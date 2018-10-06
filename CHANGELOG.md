# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [1.1.4] - 2018-09-06
### Added
- Gaussian Process Regression, Sparse Gaussian Process Regression and Students T Process Regression models
- Notebooks

## [1.1.3] - 2018-05-25
### Fixed
- HLR fit method sets shared vars if no minibatch_size given

## [1.1.2] - 2018-05-20
### Fixed
- df_summary deprecated in pymc3 release 3.3, changed to summary

## [1.1.1] - 2018-05-20
### Fixed
- Minibatches for ADVI in HLR require model_output to be cast as int

## [1.1.0] - 2018-01-30
### Added
- New class property for default number of draws for advi sampling

## [1.0.3] - 2018-01-05
### Fixed
- LICENSE file name changed to correct version
- Had to skip 1.0.2 due to PyPi uploading fiasco

## [1.0.1] - 2018-01-05
### Fixed
- Messed up uploading to PyPi

## [1.0.0] - 2018-01-05
### Added
- First version of the library 
- Hierarchical Logistic Regression and Linear Regression models
- Documentation
