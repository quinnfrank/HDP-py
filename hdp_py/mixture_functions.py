import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln as logg
from numba import jit, float64, int64, int32, boolean


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
    """Faster version of the above."""
    
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
    """Numba-compiled version of the above where New=False.  Does not support sparse matrices."""
    
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
    """Numba-compiled version of the above where new=True."""
    
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


@jit(float64[:](int64, int32[:], int32[:], int64, int64, float64[:]), nopython=True)
def cat_fk_cust4(i, x, k, Kmax, L, ha):
    """Numba-compiled version of the above where New=False.
       Only supports sparse matrices, where x is the `indices` attribute."""
    
    ll = x[i]

    ha_sum = 0
    for idx in range(L):
        ha_sum += ha[idx]
    
    # Store the size of sets V and V_l for each k
    N = x.shape[0]
    V_kks = np.zeros(Kmax)
    Vl_kks = np.zeros(Kmax)
    fk_cust = np.zeros(Kmax)
    
    for idx in range(N):
        # Compute the V and Vl sets with one pass through the data
        kk = k[idx]
        V_kks[kk] += 1
        Vl_kks[kk] += (x[idx] == ll)
    
    for kk in range(Kmax):
        fk_cust[kk] = (Vl_kks[kk] - (x[idx] == kk) + ha[ll]) / (V_kks[kk] - 1 + ha_sum)
    return fk_cust


@jit(float64(int64, int32[:], int32[:], int64, int64, float64[:]), nopython=True)
def cat_fk_cust4_new(i, x, k, Kmax, L, ha):
    """Numba-compiled version of the above where new=True."""
    
    ll = x[i]
            
    # Calculate the case where k has no members
    ha_sum = 0
    for idx in range(L):
        ha_sum += ha[idx]
        
    return ha[ll] / ha_sum