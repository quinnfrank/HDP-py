import numpy as np


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