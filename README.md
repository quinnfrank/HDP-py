# Python Implementation of Hierarchical Dirichlet Processes

Repository for final STA 663 project on Hierarchical Dirichlet Processes

This package implements the Hierarchical Dirichlet Process (HDP) described by [Teh, et al (2006)](https://www.tandfonline.com/doi/abs/10.1198/016214506000000302), a Bayesian nonparametric algorithm which can model the distribution of grouped data exhibiting clustering behavior both within and between groups.  We implement two different Gibbs samplers in Python to approximate the posterior distribution over the parameters of an HDP mixture model.  Although the HDP is flexible enough to model most any data-generating distribution, we currently support two conjugate models: (1) a Poisson and (2) a multinomial likelihood, with a gamma and Dirichlet prior over their respective parameters.  There is also a more optimized categorical model (a special case of the multinomial model).


## Installation

To install the package, navigate to this directory and use `python setup.py install`, then import the class with `from hdp_py import HDP`.

The following prerequisites are required and will be installed by default (the versions shown were those used in development):
+ `numpy` 1.16.5
+ `pandas` 0.25.1
+ `scipy` 1.3.1
+ `numba` 0.45.1

The following packages are not required but are recommended to run the helper functions in the `get_data` module.  These access pre-formatted data compatible with the HDP class and provide functionality to compare the performance of our HDP implementation with `gensim`'s LDA implementation.
+ `beautifulsoup4` 4.8.0
+ `nltk` 3.4.5
+ `gensim` 3.8.0

## Authors

+ Quinn Frank
+ Morris Greenberg
+ George Lindner

This project was made for STA 663 (Spring, 2020) at Duke University.


