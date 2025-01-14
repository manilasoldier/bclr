# <code>bclr</code>

A package for Bayesian Changepoint detection via Logistic Regression. Method was developed in Thomas, Jauch, and Matteson (2025). [Link](https://arxiv.org/abs/2401.02917)

## Installation

To install the package (as of 1/14/25), clone the repository and then install via pip (Linux/Mac):

```
$ git clone https://github.com/manilasoldier/bclr.git
$ cd bclr
$ pip install .
```

or run

```
$ pip install git+https://github.com/manilasoldier/bclr.git
```

## Information on the package

Brief tutorials (as well as replication of some of our experiemnts can be seen in <code>bclr_examples.ipynb</code> and <code>bclr_multi_examples.ipynb</code> for the single and multiple changepoint settings. The experiments folder contains the code necessary to replicate some of our experiments. However, not all of the data is available for Experiments 1 and 2 because of size constraints. It can be made available upon request.


## Running the code


```python
import bclr
import numpy as np
```


```python
np.random.seed(199203)
b1 = np.random.randn(100, 10)
b2 = np.random.randn(80, 10) + np.broadcast_to(np.random.randn(10), (80, 10))
b3 = np.random.randn(140, 10) + np.broadcast_to(np.random.randn(10), (140, 10))
b4 = np.random.randn(80, 10) + np.broadcast_to(np.random.randn(10), (80, 10))

X = np.r_[b1, b2, b3, b4]
```

First, let's specify priors for $\beta$ coefficients and specify $J = 10$ (i.e. <code>cps=10</code>). 

Then, we will go ahead and fit <code>bclr</code> to the data and estimate (predict) the changes.


```python
prior = np.diag(np.repeat(3, 10))
bclrM = bclr.MultiBayesCC(X = X, cps = 10, 
                          prior_cov=prior, warnings=False)
cps = bclrM.fit_predict()
```

Now we can look at
- where the changes are estimated to be located, 
- the posterior probability of the estimates, and 
- the normalized entropy of the segment distribution.


```python
print(cps)
```

       Location  Posterior Probability  Normalized Entropy
    0      99.0                  0.944            0.048640
    1     180.0                  1.000            0.000000
    2     320.0                  0.996            0.005354

