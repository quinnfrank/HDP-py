"""
Testing suite for the basic functionality of the hdp_py package.
Will run at installation to verify integrity of source code.
"""

from hdp_py import HDP, get_data
import numpy as np
import pandas as pd

# Test that all the get_data functions return proper items
max_docs = np.random.randint(1,20)
min_word_count = np.random.randint(1, 10)

x_nema, j_nema = get_data.get_nematode(max_docs, min_word_count)
assert isinstance(x_nema, pd.DataFrame)
assert isinstance(j_nema, np.ndarray) and len(j_nema.shape) == 1
assert x_nema.shape[0] == j_nema.shape[0]
assert len(set(x_nema.columns)) == x_nema.shape[1]
assert np.all(x_nema.dtypes == 'int')
x_nema = x_nema.to_numpy()
assert np.all([val == 0 or val == 1 for val in np.unique(x_nema)])
assert np.all(np.sum(x_nema, axis=1) == 1)
print('get_nematode() returns proper data types')

x_reut, j_reut = get_data.get_reuters(max_docs, min_word_count)
assert isinstance(x_reut, pd.DataFrame)
assert isinstance(j_reut, np.ndarray) and len(j_nema.shape) == 1
assert x_reut.shape[0] == x_reut.shape[0]
assert len(set(x_reut.columns)) == x_reut.shape[1]
x_reut = x_reut.to_numpy()
assert np.all([val == 0 or val == 1 for val in np.unique(x_reut)])
assert np.all(np.sum(x_reut, axis=1) == 1)
print('get_reuters() returns proper data types')


# Test that the samplers work as expected
L_nema = x_nema.shape[1]
hypers_nema = (L_nema, np.ones(L_nema))
hdp_nema = HDP.HDP(f='multinomial', hypers=hypers_nema)
assert not hasattr(hdp_nema, 'cfr_samples')
hdp_nema = hdp_nema.gibbs_cfr(x_nema, j_nema, iters=1)
assert hasattr(hdp_nema, 'cfr_samples')
assert hdp_nema.cfr_samples.shape == (1, x_nema.shape[0], 2)
print('HDP.cfr_samples() works as expected')
assert not hasattr(hdp_nema, 'direct_samples') and not hasattr(hdp_nema, 'beta_samples')
hdp_nema = hdp_nema.gibbs_direct(x_nema, j_nema, iters=1, Kmax=10)
assert hasattr(hdp_nema, 'cfr_samples')
assert hasattr(hdp_nema, 'direct_samples') and hasattr(hdp_nema, 'beta_samples')
assert hdp_nema.direct_samples.shape == (1, x_nema.shape[0])
assert hdp_nema.beta_samples.shape == (1, 11)
hdp_nema = hdp_nema.gibbs_direct(x_nema, j_nema, iters=1, Kmax=10, resume=True)
assert hdp_nema.direct_samples.shape == (2, x_nema.shape[0])
assert hdp_nema.beta_samples.shape == (2, 11)
print('HDP.direct_samples() works as expected')

# Check that models are equivalent
h1 = HDP.HDP(f='categorical', hypers=hypers_nema)
h2 = HDP.HDP(f='categorical_fast', hypers=hypers_nema)
np.random.seed(0)
h1 = h1.gibbs_direct(x_nema, j_nema, iters=1, Kmax=10)
np.random.seed(0)
h2 = h2.gibbs_direct(x_nema, j_nema, iters=1, Kmax=10)
assert np.allclose(h1.direct_samples, h2.direct_samples)
assert np.allclose(h1.beta_samples, h2.beta_samples)
print('HDP categorical models give equivalent results')