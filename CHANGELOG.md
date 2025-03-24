# Changelog

## v0.3.3 (3/24/25)

### Added

- Parallelization for MultiBayesCC via joblib

### Fixed

- Cleaned up code (removed whitespace, etc.)

## v0.3.2 (3/19/25)

### Added

- SNR calculation for posterior beta values.
- Helping functions to raise errors if prior mean or covariance is does not satisfy constraints of. 
- Added testing for bclr_one.py and bclr_multi.py.

### Fixed 

- Issue with conversion of np array to scalar which would cause issues in the future.

### Removed

- small_probs argument from BayesCC fit method, as there is no foreseeable use for this feature.
- random_init argument from warm_up method in MultiBayesCC, as this does not seem to have tangible benefits.
- sample_sep function from bclr_helper, because of the removal of random_init.

## v0.3.1 (1/14/25)

### Fixed

- Issue with too low of `tol` value for BayesCC `fit` method.

### Added 

- Multiple changepoint convenience functions (e.g. fit_predict, fit_transform, etc...)
