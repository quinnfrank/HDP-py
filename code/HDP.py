import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln as logg
from functools import partial
from numba import jit, float64, int64, int32


def pois_fk_cust(i, x, k, Kmax, ha, hb, new=False):
    """
    Computes the mixture components for a given customer across all k values.
    MODEL: base measure H ~ Gamma(ha, hb), F(x|phi) ~ Poisson(phi)
    All components are calculated exactly in log-space and then exponentiated.
    
    returns: (Kmax,) vector; if new=True, returns a scalar
    """
    
    x = x.flatten()  # reshape to 1D, since gibbs routine passes in a 2D array
    
    # Calculate the case where k has no members
    fknew_cust = np.exp( -logg(x[i] + 1) + logg(x[i] + ha) - logg(ha) -
                         (x[i] + ha)*np.log(1 + hb) + ha*np.log(hb) )
    if new == True: return fknew_cust        
    
    x_kks = [x[k == kk] for kk in range(Kmax)]  # subset of customers eating kk
    xi_in = np.zeros(Kmax)                      # offset if x[i] is in this subset
    xi_in[k[i]] = 1
      
    # Compute (a,b) params from gamma kernel tricks done in fk function
    av = np.array(list(map(np.sum, x_kks))) - xi_in*x[i] + ha
    bv = np.array(list(map(len, x_kks))) - xi_in + hb
    fk_cust = np.exp( -logg(x[i] + 1) + logg(x[i] + av) - logg(av) -
                      (x[i] + av)*np.log(1 + bv) + av*np.log(bv) )
     
    return fk_cust


def pois_fk_tabl(jj, tt, x, j, t, k, Kmax, ha, hb, new=False):
    """
    Computes the mixture components for a given table across all k values.
    MODEL: base measure H ~ Gamma(ha, hb), F(x|phi) ~ Poisson(phi)
    All components are calculated exactly in log-space and then exponentiated.
    
    returns: (Kmax,) vector; if new=True, returns a scalar
    """
    
    x = x.flatten()  # reshape to 1D, since gibbs routine passes in a 2D array
    x_jt = x[np.logical_and(j == jj, t == tt)]
    kk = k[np.logical_and(j == jj, t == tt)]
    
    fknew_tabl = np.exp( -np.sum(logg(x_jt + 1)) + logg(np.sum(x_jt) + ha) - logg(ha) -
                         (np.sum(x_jt) + ha)*np.log(len(x_jt) + hb) + ha*np.log(hb) )
    # If table jt doesn't exist, just return the "new" mixture component
    if len(x_jt) == 0:
        #print(f"WARNING: table {(jj, tt)} does not exist currently")
        new = True
    if new == True: return np.full(Kmax, fknew_tabl)
    
    x_kks = [x[k == kk] for kk in range(Kmax)]  # subset of customers at tables serving kk
    xjt_in = np.zeros(Kmax)                     # offset if table x_jt is in this subset
    xjt_in[kk[0]] = 1
      
    # Compute (a,b) params from gamma kernel tricks done in fk function
    av = np.array(list(map(np.sum, x_kks))) - xjt_in*np.sum(x_jt) + ha
    bv = np.array(list(map(len, x_kks))) - xjt_in*len(x_jt) + hb
    fk_tabl = np.exp( -np.sum(logg(x_jt + 1)) + logg(np.sum(x_jt) + av) - logg(av) -
                       (np.sum(x_jt) + av)*np.log(len(x_jt) + bv) + av*np.log(bv) )
     
    return fk_tabl


def mnom_fk_cust(i, x, k, Kmax, L, ha, new=False):
    """
    Computes the mixture components for a given customer across all k values.
    MODEL: base measure H ~ Dirichlet(L, ha_1,...,ha_L),
                        F(x|phi) ~ Multinomial(n_ji, phi_1,...,phi_L)
    All components are calculated exactly in log-space and then exponentiated.
    X can be a dense or a sparse csr-style matrix.
    
    returns: (Kmax,) vector; if new=True, returns a scalar
    """
    
    xi, ni = x[i, :], np.sum(x[i, :])
    log_con = logg(ni + 1) - np.sum(logg(xi + np.ones(L))) # term constant for all k
    # Calculate the case where k has no members
    
    if new == True:
        fknew_cust = np.exp( log_con + np.sum(logg(xi + ha)) - logg(np.sum(xi + ha)) + 
                             logg(np.sum(ha)) - np.sum(logg(ha)) )
        return fknew_cust        
    
    # Get subset of customers eating kk; each entry is a (#, L) matrix
    x_kks = [x[k == kk, :] for kk in range(Kmax)]  
    
    # Compute params from Dirichlet kernel tricks done in fk function
    a_bot = np.vstack([np.sum(x_kk, axis=0) for x_kk in x_kks]) + ha[None, :]    # (Kmax, L)
    a_bot[k[i], :] -= xi                         # offset if xi is in this subset
    a_top = np.apply_along_axis(lambda row: row + xi, 1, a_bot)
    fk_cust = np.exp( log_con + np.sum(logg(a_top), axis=1) - logg(np.sum(a_top, axis=1)) +
                      logg(np.sum(a_bot, axis=1)) - np.sum(logg(a_bot), axis=1) )
     
    # Convert back to a dense array in case X was sparse
    return np.asarray(fk_cust).ravel()


def mnom_fk_tabl(jj, tt, x, j, t, k, Kmax, L, ha, new=False):
    """
    Computes the mixture components for a given customer across all k values.
    MODEL: base measure H ~ Dirichlet(L, ha_1,...,ha_L),
                        F(x|phi) ~ Multinomial(n_ji, phi_1,...,phi_L)
    All components are calculated exactly in log-space and then exponentiated.
    
    returns: (Kmax,) vector; if new=True, returns a scalar
    """
    
    x_jt = x[np.logical_and(j == jj, t == tt), :]                                # (|T|, L)
    kk = k[np.logical_and(j == jj, t == tt)]
    n_jt = np.sum(x_jt, axis=1)                                                  # (|T|,)
    sum_jt = np.sum(x_jt, axis=0)                                                # (L,)
    log_con = np.sum(logg(n_jt + 1)) - np.sum(logg(x_jt + 1))    # term constant for all k
    
    fknew_tabl = np.exp( log_con + np.sum(logg(sum_jt + ha)) - logg(np.sum(sum_jt + ha)) + 
                         logg(np.sum(ha)) - np.sum(logg(ha)) )
    # If table jt doesn't exist, just return the "new" mixture component
    if x_jt.shape[0] == 0:
        #print(f"WARNING: table {(jj, tt)} does not exist currently")
        new = True
    if new == True: return fknew_tabl       
    
    # Get subset of customers eating kk; each entry is a (#, L) matrix
    x_kks = [x[k == kk, :] for kk in range(Kmax)]
      
    # Compute params from Dirichlet kernel tricks done in fk function
    a_bot = np.vstack([np.sum(x_kk, axis=0) for x_kk in x_kks]) + ha[None, :]    # (Kmax, L)
    a_bot[kk[0], :] -= sum_jt                       # offset if table x_jt is in this subset
    a_top = a_bot + sum_jt[None, :]
    fk_tabl = np.exp( log_con + np.sum(logg(a_top), axis=1) - logg(np.sum(a_top, axis=1)) +
                      logg(np.sum(a_bot, axis=1)) - np.sum(logg(a_bot), axis=1) )

    return fk_tabl


def cat_fk_cust(i, x, k, Kmax, L, ha, new=False):
    """
    Computes the mixture components for a given customer across all k values.
    MODEL: base measure H ~ Dirichlet(L, ha_1,...,ha_L),
                        F(x|phi) ~ Categorical(L, phi_1,...,phi_L)
    All components are calculated exactly in log-space and then exponentiated.
    X can be a dense or a sparse csr-style matrix.
    
    returns: (Kmax,) vector; if new=True, returns a scalar
    """
    
    xi = x[i, :]
    ll = sparse.find(xi)[1][0]        # get column index of the 1 value
    # Calculate the case where k has no members
    if new == True:
        return ha[ll] / np.sum(ha)    
    
    # Store the size of sets V and V_l for each k
    V_kks = np.array([np.sum(k == kk) for kk in range(Kmax)])
    Vl_kks = np.array([np.sum(x[k == kk, ll]) for kk in range(Kmax)])
    
    fk_cust = (Vl_kks + ha[ll]) / (V_kks + np.sum(ha))
    return fk_cust


def cat_fk_cust2(i, x, k, Kmax, L, ha, new=False):
    """
    Faster version of the above.
    """
    
    xi = x[i, :]
    ll = sparse.find(xi)[1][0]        # get column index of the 1 value
    # Calculate the case where k has no members
    if new == True:
        return ha[ll] / np.sum(ha)    
    
    # Store the size of sets V and V_l for each k
    V_kks = np.zeros(Kmax)
    kk_counts = pd.Series(k).value_counts()
    V_kks[kk_counts.index] = kk_counts
    Vl_kks = np.array([np.sum(x[k == kk, ll]) for kk in range(Kmax)])
    
    fk_cust = (Vl_kks + ha[ll]) / (V_kks + np.sum(ha))
    return fk_cust


@jit(float64[:](int64, int32[:,:], int32[:], int64, int64, float64[:]), nopython=True)
def cat_fk_cust3(i, x, k, Kmax, L, ha):
    """
    Numba-compiled version of the above.
    """
    
    ll = 0                           # get column index of the 1 value
    for idx in range(L):
        if x[i, idx] == 1:
            ll = idx
            break
    
    ha_sum = 0
    for idx in range(L):
        ha_sum += ha[idx]
    
    # Store the size of sets V and V_l for each k
    V_kks = np.zeros(Kmax)
    Vl_kks = np.zeros(Kmax) 
    fk_cust = np.zeros(Kmax)
    N = x.shape[0]
    for kk in range(Kmax):
        # Compute a mask which gives the i indices of observations with value k
        for idx in range(N):
            if k[idx] == kk:
                V_kks[kk] += 1
                Vl_kks[kk] += x[idx, ll]
        fk_cust[kk] = (Vl_kks[kk] + ha[ll]) / (V_kks[kk] + ha_sum)
    
    return fk_cust


@jit(float64(int64, int32[:,:], int32[:], int64, int64, float64[:]), nopython=True)
def cat_fk_cust3_new(i, x, k, Kmax, L, ha):
    """
    Numba-compiled version of the above.
    """
    
    ll = 0                           # get column index of the 1 value
    for idx in range(L):
        if x[i, idx] == 1:
            ll = idx
            break
            
    # Calculate the case where k has no members
    ha_sum = 0
    for idx in range(L):
        ha_sum += ha[idx]
        
    return ha[ll] / ha_sum
        

########################################################################################


class StirlingEngine:
    """
    Numerically efficient engine for computing and storing computed Stirling numbers.
    
    CONSTRUCTOR PARAMETERS
    - Nmax: largest integer n for which s(n,m) will need to be computed
    
    PRIVATE ATTRIBUTES
    - s_memo_, slog_memo_: running tables of previously computed values
    """
    
    def __init__(self, Nmax):
        self.s_memo_ = np.full((Nmax, Nmax), np.nan)
        self.slog_memo_ = np.full((Nmax, Nmax), np.nan)
        
        
    def stirling(self, n, m):
        """
        Computes an unsigned Stirling number of the first kind.
        Uses dynamic programming to store previously computed s(n,m) values,
        as this is a repeatedly-called recursive algorithm.
        """
        
        assert n < self.s_memo_.shape[0] and m < self.s_memo_.shape[0]  
        # If this has already been computed, return stored value
        if not np.isnan(self.s_memo_[n, m]):
            return self.s_memo_[n, m]
        else:
            return_val = np.nan

            # Base cases
            if (n == 0 and m == 0) or (n == 1 and m == 1):
                return_val = 1
            elif (n > 0 and m == 0) or m > n:
                return_val = 0
            # Recursion relation
            else:
                return_val = self.stirling(n-1, m-1) + (n-1)*self.stirling(n-1, m)

            self.s_memo_[n, m] = return_val
            return return_val
    
    
    def stirlog(self, n, m):
        """
        Computes the natural logarithm of an unsigned Stirling number,
        using the same dynamic programming approach as above.
        If s(n,m) = 0, this gets returned as -inf (np.exp(-inf) == 0.0)
        
        This is the preferred function, as stirling() can encounter overflow errors.
        """
        
        assert n < self.slog_memo_.shape[0] and m < self.slog_memo_.shape[0]  
        # If this has already been computed, return stored value
        if not np.isnan(self.slog_memo_[n, m]):
            return self.slog_memo_[n, m]
        else:
            return_val = np.nan

            # Base cases
            if (n == 0 and m == 0) or (n == 1 and m == 1):
                return_val = 0
            elif (n > 0 and m == 0) or m > n:
                return_val = -np.inf
            # Recursion relation
            else:
                log_s1, log_s2 = self.stirlog(n-1, m-1), self.stirlog(n-1, m)
                # If s1 == 0 (log_s1 == -inf), just return (n-1)*log_s2
                # By definition, must have s2 > s1, so only need to check s1
                if np.isfinite(log_s1):
                    val = (n-1) * np.exp(log_s2 - log_s1)
                    # If there is overflow/underflow in `val`, approximate log(1+x) = log(x)
                    if np.isfinite(val):
                        return_val = log_s1 + np.log1p(val)
                    else:
                        return_val = log_s2 + np.log(n-1)
                else:
                    return_val = log_s2 + np.log(n-1)

            self.slog_memo_[n, m] = return_val
            return return_val
        

########################################################################################


class HDP:
    """
    Model implementing the Chinese Restaurant Franchise Process formulation of the HDP.
    
    CONSTRUCTOR PARAMETERS
    - gamma, alpha0: scaling parameters > 0 for base measures H and G0
    - f: string representing distribution of data; h is chosen to be conjugate
    - hypers: tuple of hyperparameter values specific to f/h scheme chosen
    
    PRIVATE ATTRIBUTES (volatile)
    - tk_map_: (J x Tmax) matrix of k values for each (j,t) pair
    - beta_: (Kmax + 1,) vector of beta values for each k
    - n_: (J x Tmax) matrix specifying counts of customers (gibbs_cfr)
    - q_: (J x Kmax) matrix specifying counts of customers (gibbs_direct)
    - m_: (J x Kmax) matrix specifying counts of tables
    - fk_cust_, fk_tabl_: functions to compute mixing components for Gibbs sampling
    - stir_: an object of class StirlingEngine which computes Stirling numbers
    
    PUBLIC ATTRIBUTES
    cfr_samples: (S x N x 2) matrix of (t, k) values for each data point i;
                 exists only after gibbs_cfr() has been called
    direct_samples: (S x N) matrix of k values for each data point i;
                    exists only after gibbs_direct() has been called
    beta_samples: (S x Kmax+1) matrix of beta values after each iteration;
                  exists only after gibbs_direct() has been called
    """
    
    def __init__(self, gamma=1, alpha0=1, f='multinomial', hypers=None):
        self.g_ = gamma
        self.a0_ = alpha0
        self.set_priors(f, hypers)
        
    def set_priors(self, f, hypers):
        """
        Initializes the type of base measure h_ and data-generation function f_.
        Also sets hypers_, the relevelant hyperparameters and
                  fk_routine_, the function to compute mixing components.
        """
        if f == 'poisson':
            # Specify parameters of H ~ Gamma(a,b)
            if hypers is None:
                self.hypers_ = (1,1)
            else: self.hypers_ = hypers
            self.fk_cust_ = pois_fk_cust
            self.fk_cust_new_ = partial(pois_fk_cust, new=True)
            self.fk_tabl_ = pois_fk_tabl
        
        elif f == 'multinomial':
            if hypers is None:
                L = 2
                self.hypers_ = (L, np.ones(L))
            else: self.hypers_ = hypers
            self.fk_cust_ = mnom_fk_cust
            self.fk_cust_new_ = partial(mnom_fk_cust, new=True)
            self.fk_tabl_ = mnom_fk_tabl
            
        elif f == 'categorical':
            # Identical to multinomial, but with some efficiency upgrades
            if hypers is None:
                L = 2
                self.hypers_ = (L, np.ones(L))
            else: self.hypers_ = hypers
            self.fk_cust_ = cat_fk_cust
            self.fk_cust_new_ = partial(cat_fk_cust, new=True)
            self.fk_tabl_ = mnom_fk_tabl
            
        elif f == 'categorical_numba':
            # Identical to multinomial, but with some efficiency upgrades
            if hypers is None:
                L = 2
                self.hypers_ = (L, np.ones(L))
            else:
                # Ensure hyperparameters have proper data types, for numba functions
                self.hypers_ = (int(hypers[0]), hypers[1].astype('float'))
            self.fk_cust_ = cat_fk_cust3
            self.fk_cust_new_ = cat_fk_cust3_new
            self.fk_tabl_ = mnom_fk_tabl

    
    def tally_up(self, it, which=None):
        """
        Helper function for computing maps and counts in gibbs().
        Given a current iteration in the cfr_samples attribute, does a full
        recount of customer/table allocations, updating n_ and m_.
        Set which = 'n' or 'm' to only tally up that portion
        """    
        
        if which == 'n':
            jt_pairs = self.cfr_samples[it,:,0:2]
            # Count customers at each table (jt)
            cust_counts = pd.Series(map(tuple, jt_pairs)).value_counts()
            j_idx, t_idx = tuple(map(np.array, zip(*cust_counts.index)))
            self.n_ *= 0
            self.n_[j_idx, t_idx] = cust_counts
            
        elif which == 'm':
            jt_pairs = self.cfr_samples[it,:,0:2]
            # First filter by unique tables (jt), then count tables with each k value
            jt_unique, k_idx = np.unique(jt_pairs, axis=0, return_index=True)
            jk_pairs = np.c_[self.cfr_samples[it, k_idx, 0],
                             self.cfr_samples[it, k_idx, 2]]
            #print(jk_pairs)
            tabl_counts = pd.Series(map(tuple, jk_pairs)).value_counts()
            #print(tabl_counts)
            j_idx, k_idx = tuple(map(np.array, zip(*tabl_counts.index)))
            self.m_ *= 0
            self.m_[j_idx, k_idx] = tabl_counts
            
        elif which == 'q':
            jk_pairs = self.direct_samples[it,:,:]
            # Counts customers at each j eating k
            cust_counts = pd.Series(map(tuple, jk_pairs)).value_counts()
            j_idx, k_idx = tuple(map(np.array, zip(*cust_counts.index)))
            self.q_ *= 0
            self.q_[j_idx, k_idx] = cust_counts
            
    
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
    
    
    def draw_t(self, it, x, j, Tmax, Kmax, verbose):
        """
        Helper function which does the draws from the t_ij full conditional.
        Updates the counts and the samples matrices at iteration `it`.
        Called by gibbs_cfr()
        """
        
        t_next, k_next = self.cfr_samples[it,:,1], self.cfr_samples[it,:,2]
        # Cycle through the t value of each customer, conditioning on everything
        # Randomize the order in which updates occur
        for i in np.random.permutation(len(j)):
            jj, tt0, kk0 = j[i], t_next[i], k_next[i]

            # Get vector of customer f_k values (dependent on model specification)
            old_mixes = self.fk_cust_(i, x, k_next, Kmax, *self.hypers_) 
            new_mixes = self.fk_cust_(i, x, k_next, Kmax, *self.hypers_, new=True) 
            # Calculate pointwise likelihoods p(x_ji | ...)
            M = np.sum(self.m_)
            Mk = np.sum(self.m_, axis=0)   # number of tables serving k
            lik = old_mixes @ (Mk / (M + self.g_)) + new_mixes * (self.g_ / (M + self.g_))

            cust_offset = np.zeros(Tmax)
            cust_offset[tt0] = 1
            old_t = (self.n_[jj, :] - cust_offset) * old_mixes[self.tk_map_[jj, :]]      
            new_t = self.a0_ * lik
            # If a table is in use, prob comes from old_t; otherwise, from new_t
            # Distribute the weight of new_t across all possible new allocations
            t_used = self.n_[jj, :] > 0
            t_dist = self.get_dist(old_t, new_t, t_used, Tmax)

            tt1 = np.random.choice(Tmax, p=t_dist)
            t_next[i] = tt1
            self.tally_up(it, which='n')

            # If this table was previously unoccupied, we need to select a k
            if self.n_[jj, tt1] == 1 and tt0 != tt1:
                old_k = np.sum(self.m_, axis=0) * old_mixes
                new_k = self.g_ * new_mixes
                k_used = np.sum(self.m_, axis=0) > 0
                k_dist = self.get_dist(old_k, new_k, k_used, Kmax)

                kk1 = np.random.choice(Kmax, p=k_dist)
                self.tk_map_[jj, tt1] = kk1
                k_next[i] = self.tk_map_[jj, tt1]
            self.tally_up(it, which='m')
    
    
    def draw_k(self, it, x, j, Kmax, verbose):
        """
        Helper function which does the draws from the t_ij full conditional.
        Updates the counts and the samples matrices at iteration `it`.
        Called by gibbs_cfr()
        """
        
        t_next, k_next = self.cfr_samples[it,:,1], self.cfr_samples[it,:,2]
        # Cycle through the k values of each table
        j_idx, t_idx = np.where(self.n_ > 0)   # find the occupied tables
        for i in np.random.permutation(len(j_idx)):
            jj, tt = j_idx[i], t_idx[i]
            kk0 = self.tk_map_[jj, tt]

            # Get vector of table f_k values (dependent on model specification)
            old_mixes = self.fk_tabl_(jj, tt, x, j, t_next, k_next, Kmax, *self.hypers_) 
            new_mixes = self.fk_tabl_(jj, tt, x, j, t_next, k_next, Kmax, *self.hypers_, new=True) 

            tabl_offset = np.zeros(Kmax)
            tabl_offset[kk0] = 1
            old_k = (np.sum(self.m_, axis=0) - tabl_offset) * old_mixes
            new_k = self.g_ * new_mixes
            k_used = np.sum(self.m_, axis=0) > 0
            k_dist = self.get_dist(old_k, new_k, k_used, Kmax)

            kk1 = np.random.choice(Kmax, p=k_dist)
            self.tk_map_[jj, tt] = kk1
            k_next[np.logical_and(j == jj, t_next == tt)] = kk1
            self.tally_up(it, which='m')
    
    
    def draw_z(self, it, x, j, Kmax, verbose):
        """
        Helper function which does the draws from the z_ij full conditional.
        Updates the counts and the samples matrices at iteration `it`.
        Called by gibbs_direct()
        """
        
        k_next = self.direct_samples[it,:,1]
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
            k_used = np.sum(self.m_, axis=0) > 0
            k_dist = self.get_dist(old_k, new_k, k_used, Kmax)

            kk1 = np.random.choice(Kmax, p=k_dist)
            k_next[i] = kk1
            self.q_[jj, kk0] -= 1
            self.q_[jj, kk1] += 1
            
            # If this k value was previously unused, must also set the beta_k component
            if np.sum(self.q_[:, kk1] == 1):
                b = np.random.beta(1, self.g_)
                beta_u = self.beta_samples[it, -1]
                self.beta_samples[it, kk1] = b * beta_u
                self.beta_samples[it, -1] = (1-b) * beta_u
                
    
    def draw_m(self, it, x, j, Kmax, verbose):
        """
        Helper function which does the draws from the z_ij full conditional.
        Updates the counts and the samples matrices at iteration `it`.
        Called by gibbs_direct()
        """
        
        k_next = self.direct_samples[it,:,1]
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
                
    
    def gibbs_cfr(self, x, j, iters, Tmax=None, Kmax=None, verbose=False):
        """
        Runs the Gibbs sampler to generate posterior estimates of t and k.
        x: data matrix, stored row-wise if multidimensional
        j: vector of group labels; must have same #rows as x
        iters: number of iterations to run
        Tmax: maximum number of clusters for each group
        Kmax: maximum number of atoms to draw from base measure H
        
        returns: this HDP object with cfr_samples attribute
        """
        
        group_counts = pd.Series(j).value_counts()
        J, N = np.max(j) + 1, len(j)
        # Set default Tmax and Kmax, if not provided
        if Tmax is None: Tmax = min(100, np.max(group_counts))
        if Kmax is None: Kmax = min(100, N)
            
        self.n_ = np.zeros((J, Tmax), dtype='int')
        self.m_ = np.zeros((J, Kmax), dtype='int')
        self.cfr_samples = np.zeros((iters+1, N, 3), dtype='int')
        self.cfr_samples[:,:,0] = j
        np.seterr('ignore')
        
        # Set random initial values for t and k assignments
        t0, k0 = self.cfr_samples[0,:,1], self.cfr_samples[0,:,2]
        t0[:] = np.random.randint(0, Tmax, size=N)
        self.tk_map_ = np.random.randint(0, Kmax//2, (J, Tmax))
        self.tally_up(it=0, which='n')
        for jj in range(J):
            for tt in np.where(self.n_[jj, :] > 0)[0]:
                k0[np.logical_and(j == jj, t0 == tt)] = self.tk_map_[jj, tt]
        self.tally_up(it=0, which='m')
        
        for s in range(iters):
            t_prev, k_prev = self.cfr_samples[s,:,1], self.cfr_samples[s,:,2]
            t_next, k_next = self.cfr_samples[s+1,:,1], self.cfr_samples[s+1,:,2]
            # Copy over the previous iteration as a starting point
            t_next[:], k_next[:] = t_prev, k_prev
            
            self.draw_t(s+1, x, j, Tmax, Kmax, verbose)
            self.draw_k(s+1, x, j, Kmax, verbose)
        
        self.cfr_samples = self.cfr_samples[1:,:,1:]
        return self  
    
    
    def gibbs_direct(self, x, j, iters, Kmax=None, resume=False, verbose=False):
        """
        Runs the Gibbs sampler to generate posterior estimates of k.
        x: data matrix, stored row-wise if multidimensional
        j: vector of group labels; must have same #rows as x
        iters: number of iterations to run
        Kmax: maximum number of atoms to draw from base measure H
        resume: if True, will continue from end of previous direct_samples, if dimensions match up
        
        returns: this HDP object with direct_samples attribute
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
        
        self.direct_samples = np.zeros((iters+1, N, 2), dtype='int')
        self.direct_samples[:,:,0] = j
        self.beta_samples = np.zeros((iters+1, Kmax+1))
        self.stir_ = StirlingEngine(np.max(group_counts) + 1)
        np.seterr('ignore')
        
        if resume == True:
            # Fill in the start of the samples with the previously computed samples
            self.direct_samples[1:start+1,:,1] = prev_direct
            self.beta_samples[1:start+1,:] = prev_beta
            # q_ and m_ attributes should already still exist within the object
        else:
            self.q_ = np.zeros((J, Kmax), dtype='int')   # performs the same function as n_
            self.m_ = np.zeros((J, Kmax), dtype='int')
            
            # Set random initial values for k assignments
            k0 = self.direct_samples[0,:,1]
            k0[:] = np.random.randint(0, Kmax, size=N)
            self.tally_up(it=0, which='q')
            # Implicitly set random t assignments by drawing possible m counts (m_jk <= q_jk)
            for jj in range(J):
                for kk in range(Kmax):
                    max_m = self.q_[jj, kk]
                    if max_m == 1:
                        self.m_[jj, kk] = 1
                    elif max_m > 1:
                        self.m_[jj, kk] = np.random.randint(1, max_m)
            # Compute the corresponding beta values from m assignments
            Mk = np.sum(self.m_, axis=0)
            self.beta_samples[0,:] = np.random.dirichlet(np.append(Mk, self.g_) + 1e-10)
        
        for s in range(start, iters):
            # Copy over the previous iteration as a starting point
            self.direct_samples[s+1,:,1] = self.direct_samples[s,:,1] 
            self.beta_samples[s+1,:] = self.beta_samples[s,:]
            
            self.draw_z(s+1, x, j, Kmax, verbose)
            self.draw_m(s+1, x, j, Kmax, verbose)
            
            Mk = np.sum(self.m_, axis=0)
            # Dirichlet weights must be > 0, so in case some k is unused, add epsilon
            self.beta_samples[s+1,:] = np.random.dirichlet(np.append(Mk, self.g_) + 1e-10)
            if verbose: print(self.beta_samples[s+1,:].round(3))
        
        self.direct_samples = self.direct_samples[1:,:,1]
        self.beta_samples = self.beta_samples[1:,:]
        return self
        