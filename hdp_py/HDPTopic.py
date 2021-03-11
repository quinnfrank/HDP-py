import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln as logg
from functools import partial
from numba import jit, float64, int64, int32, boolean

from hdp_py.StirlingEngine import *
from hdp_py.mixture_functions import *


class HDPTopic:
    """
    Model implementing the Chinese Restaurant Franchise Process formulation of the HDP.
    
    CONSTRUCTOR PARAMETERS
    - vocab: a gensim Dictionary object which maps integer ids -> tokens
    - h_con: |vocab| length vector of concentration parameters for Dirichlet prior
    - gamma, alpha0: scaling parameters > 0 for base measures H and G0
    
    
    PRIVATE ATTRIBUTES (volatile)
    - q_: (J x Kmax) matrix specifying counts of customers (gibbs_direct)
    - m_: (J x Kmax) matrix specifying counts of tables
    - fk_cust_, fk_cust_new_: functions to compute mixing components for Gibbs sampling
    - stir_: an object of class StirlingEngine which computes Stirling numbers
    
    PUBLIC ATTRIBUTES
    - direct_samples: (S x N) matrix of k values for each data point i;
                      exists only after gibbs_direct() has been called
    - beta_samples: (S x Kmax+1) matrix representing topic distribution over corpus
                    exists only after gibbs_direct() has been called
    - pi_samples: (S x J x Kmax+1) matrix representing topic distribution over each document
                  exists only after gibbs_direct() has been called
    """
    
    def __init__(self, vocab, h_con=None, gamma=1, alpha0=1):
        self.g_ = gamma
        self.a0_ = alpha0
        # Set a weak, uniform concentration if none provided
        if h_con is None:
            L = len(vocab)
            self.hypers_ = (L, np.ones(L))
        else:
            self.hypers_ = (len(h_con), h_con)
        
        self.vocab = vocab
        self.fk_cust_ = cat_fk_cust4
        self.fk_cust_new_ = cat_fk_cust4_new

        
    def get_dist(self, old, new, used, size):
        """
        Helper function which standardizes the operation of computing a
        full conditional distribution, for both t and k values.
        Also normalizes and ensures there are no NANs.
        - old: a (size,) vector of probability values for used values
        - new: a scalar representing the combined probability of all unused values
        - used: a (size,) mask encoding which values in the sample space are being used
        - size: the size of the sample space
        """

        num_unused = size - np.sum(used)
        dist = None
        if num_unused == 0:
            # In our truncated sample space, there is no room for "new" values
            dist = old
        else:
            dist = old * used + (new / num_unused) * np.logical_not(used)
        
        # Remove nans and add epsilon so that distribution is all positive
        #print(f"{dist.round(3)} (sum = {np.sum(dist)})")
        dist[np.logical_not(np.isfinite(dist))] = 0
        dist += 1e-10
        return dist / np.sum(dist)
    
    
    @staticmethod
    @jit(float64[:](float64[:], float64, boolean[:], int32), nopython=True)
    def get_dist_compiled(old, new, used, size):
        """
        Helper function which standardizes the operation of computing a
        full conditional distribution, for both t and k values.
        Also normalizes and ensures there are no NANs.
        This version is compiled via numba to make repeated calls more efficient.
        - old: a (size,) vector of probability values for used values
        - new: a scalar representing the combined probability of all unused values
        - used: a (size,) mask encoding which values in the sample space are being used
        - size: the size of the sample space
        """
        
        dist = np.zeros(size)
        dist_sum, num_unused = 0, 0
        # Spread out the probability of a "new" value over all unused slots
        for k in range(size):
            if not used[k]:
                num_unused += 1
        
        for k in range(size):
            dist[k] = old[k] if used[k] else new / num_unused
            if not np.isfinite(dist[k]): dist[k] = 0
            dist[k] += 1e-10
            dist_sum += dist[k]
            
        # Return normalized distribution vector
        for k in range(size):
            dist[k] /= dist_sum
        return dist
    
    
    def draw_z(self, it, x, j, Kmax, verbose):
        """
        Helper function which does the draws from the z_ij full conditional,
        assigning clusters to each individual token.
        Updates the counts and the samples matrices at iteration `it`.
        Called by gibbs_direct()
        """
        
        k_next = self.direct_samples[it,:]
        k_counts = np.sum(self.q_, axis=0)
        
        # Cycle through the k values of each customer
        for i in np.random.permutation(len(j)):
            jj, kk0 = j[i], k_next[i]
            
            # Get vector of customer f_k values (dependent on model specification)
            old_mixes = self.fk_cust_(i, x, k_next, Kmax, *self.hypers_) 
            new_mixes = self.fk_cust_new_(i, x, k_next, Kmax, *self.hypers_) 
            
            cust_offset = np.zeros(Kmax)
            cust_offset[kk0] = 1
            old_k = (self.q_[jj, :] - cust_offset +
                     self.a0_ * self.beta_samples[it, :-1]) * old_mixes      
            new_k = self.a0_ * self.beta_samples[it, -1] * new_mixes
            k_used = k_counts > 0
            k_dist = self.get_dist_compiled(old_k, new_k, k_used, Kmax)
            
            kk1 = np.random.choice(Kmax, p=k_dist)
            # Update the necessary count vectors
            k_next[i] = kk1
            self.q_[jj, kk0] -= 1
            k_counts[kk0] -= 1
            self.q_[jj, kk1] += 1
            k_counts[kk1] += 1
            
            # If kk0 is now unused, must remove its beta_k component and re-normalize
            if k_counts[kk0] == 0:
                self.beta_samples[it, kk0] = 0
                self.beta_samples[it, :] /= np.sum(self.beta_samples[it, :])
            # If kk1 was previously unused, must set a new beta_k component
            if k_counts[kk1] == 1:
                b = np.random.beta(1, self.g_)
                beta_u = self.beta_samples[it, -1]
                self.beta_samples[it, kk1] = b * beta_u
                self.beta_samples[it, -1] = (1-b) * beta_u
                
    
    def draw_m(self, it, x, j, Kmax, verbose):
        """
        Helper function which does the draws from the m_jt full conditional,
        which implicitly determines the overall clustering structure for each document.
        Updates the counts and the samples matrices at iteration `it`.
        Called by gibbs_direct()
        """
        
        k_next = self.direct_samples[it,:]
        self.m_ *= 0                           # reset the m counts
        # Cycle through the k values of each restaurant
        j_idx, k_idx = np.where(self.q_ > 0)   # find the consumed dishes
        for i in np.random.permutation(len(j_idx)):
            jj, kk = j_idx[i], k_idx[i]
            max_m = self.q_[jj, kk]
            
            abk = self.a0_ * self.beta_samples[it, kk]
            m_range = np.arange(max_m) + 1
            log_s = np.array([self.stir_.stirlog(max_m, m) for m in m_range])
            m_dist = np.exp( logg(abk) - logg(abk + max_m) +
                             log_s + m_range * np.log(abk) )
            """MOSTLY FIXED.  m_dist should be a proper distribution"""
            m_dist[np.logical_not(np.isfinite(m_dist))] = 0
            m_dist += 1e-10
            
            mm1 = np.random.choice(m_range, p=m_dist/np.sum(m_dist))
            self.m_[jj, kk] = mm1
    
    
    def gibbs_direct(self, x, j, iters, Kmax=None, resume=False, verbose=False):
        """
        Runs the Gibbs sampler to generate posterior estimates of k.
        x: vector of ints mapping to individual tokens, in a bag of words format
        j: vector of ints labeling document number for each token
        iters: number of iterations to run
        Kmax: maximum number of clusters that can be used (set much higher than needed)
        resume: if True, will continue from end of previous direct_samples, if dimensions match up
        
        returns: this HDPTopic object with direct_samples attribute
        """
        
        group_counts = pd.Series(j).value_counts()
        J, N = np.max(j) + 1, len(j)
        if Kmax is None: Kmax = min(100, N)
        
        prev_direct, prev_beta = None, None
        start = 0
        if resume == True:
            # Make sure the x passed in is the same size as it previously was
            assert (N == self.direct_samples.shape[1] and
                    Kmax == self.beta_samples.shape[1] - 1), "Cannot resume with different data."
            iters += self.direct_samples.shape[0]
            prev_direct, prev_beta = self.direct_samples, self.beta_samples
            start = self.direct_samples.shape[0]
        
        self.direct_samples = np.zeros((iters+1, N), dtype='int')
        self.beta_samples = np.zeros((iters+1, Kmax+1))
        self.stir_ = StirlingEngine(np.max(group_counts) + 1)
        np.seterr('ignore')
        
        if resume == True:
            # Fill in the start of the samples with the previously computed samples
            self.direct_samples[1:start+1,:] = prev_direct
            self.beta_samples[1:start+1,:] = prev_beta
            # q_ and m_ attributes should already still exist within the object
        else:
            self.q_ = np.zeros((J, Kmax), dtype='int')   # counts number of tokens in doc j, cluster k
            self.m_ = np.zeros((J, Kmax), dtype='int')   # counts number of k-clusters in doc j
            
            # Initialize all tokens in the same cluster: q_jk = |doc_j|
            self.q_[group_counts.index, 0] = group_counts.values
            # Initialize to one cluster per document: m_jk = 1
            self.m_[:, :] = (self.q_ > 0).astype('int')
            # Compute the corresponding beta values from m assignments
            Mk = np.sum(self.m_, axis=0)
            self.beta_samples[0,:] = np.random.dirichlet(np.append(Mk, self.g_) + 1e-100)
        
        for s in range(start, iters):
            # Copy over the previous iteration as a starting point
            self.direct_samples[s+1,:] = self.direct_samples[s,:] 
            self.beta_samples[s+1,:] = self.beta_samples[s,:]
            
            self.draw_z(s+1, x, j, Kmax, verbose)
            self.draw_m(s+1, x, j, Kmax, verbose)
            
            Mk = np.sum(self.m_, axis=0)
            # Dirichlet weights must be > 0, so in case some k is unused, add epsilon
            self.beta_samples[s+1,:] = np.random.dirichlet(np.append(Mk, self.g_) + 1e-100)
            if verbose: print(self.beta_samples[s+1,:].round(3))
        
        self.direct_samples = self.direct_samples[1:,:]
        self.beta_samples = self.beta_samples[1:,:]
        return self
        