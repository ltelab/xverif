#!/usr/bin/env python3
"""
Created on Sat Feb 27 15:51:43 2021.

@author: ghiggi
"""
# https://github.com/xarray-contrib/xskillscore/tree/main/xskillscore/core


# https://github.com/pySTEPS/pysteps/tree/master/pysteps/verification
# https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/interface.pySTEPS

# Preprocessing
# https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/utils.py

# Interfaces
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/interface.py

####--------------------------------------------------------------------------.
#################
#### Metrics ####
#################
# Deterministic continuous
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/deterministic.py (xarray)
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcontscores.py
# - https://xskillscore.readthedocs.io/en/stable/api/xskillscore.pearson_r_eff_p_value.html
# - https://github.com/hzambran/hydroGOF/tree/master/R
# --> Tests: https://github.com/pySTEPS/pysteps/blob/master/pysteps/tests/test_verification_detcontscores.py

# Deterministic categorical
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/contingency.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcatscores.py
# --> Tests: https://github.com/pySTEPS/pysteps/blob/master/pysteps/tests/test_verification_detcatscores.py

# Probabilistic continuous
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/probabilistic.py
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_probabilistic.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/ensscores.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/probscores.py

# Probabilistic categorical

# Distribution metrics

# Spatial metrics
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/salscores.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/spatialscores.py

# Hypothesis tests
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/stattests.py
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/comparative.py

# Effective sample size
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L109

# resampling/bootstrapping:
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/resampling.py

####--------------------------------------------------------------------------.
#### Preprocessing by chunk
# - Point/Pixel-wise
#   --> Preprocessing apply to each pixel timeseries separetely
#   --> ds_forecast.chunk({"time": -1}

# - Spatial chunks
#   --> Preprocessing apply to spatial region
#   --> If given timestep only one pixel nan (in obs or pred), all timesteps becomes nan
#   --> Or metrics deals with np.nan (but not efficient?)
#   --> Or looping over dimensions with numba and avoid vectorization?


####--------------------------------------------------------------------------.
#### Dimension names

## aggregating_dims --> TODO argument name
# - check available in both datasets
# - time if want to calculate i.e. pixelwise the skills
# - x, y if want to calculate overall metrics at each timestep

## support_dims
# - dims present in both dataset which are not aggregating dims

## broadcast_dims
# - dims present in only 1 dataset
# - If obs to be broadcasted on pred (then become support_dims)

####--------------------------------------------------------------------------.
##############
#### TODO ####
##############
#### - Defaults choice
# - aggregating_dim --> None?
# - Change name of aggregating_dim
# - Change name of sample and aux dimensions

####--------------------------------------------------------------------------.
#### Daskify functions


####--------------------------------------------------------------------------.
#### Numba functions


####--------------------------------------------------------------------------.
#### - Suppress warnings
# - Division by 0

# RuntimeWarning: Degrees of freedom <= 0 for slice   (when all np.nan in nanstd)

# suppress_warnings("All-NaN slice encountered")
# with suppress_warnings("invalid value encountered in true_divide"):
#         with suppress_warnings("invalid value encountered in double_scalars"):

####--------------------------------------------------------------------------.
#### - Categorical verification
# --> Binary or multiclass (separated)
# --> If probababilities, add tool to predict classes using various probabilities (--> prob_threshold dimension)
# --> Expect classes (0,1) or (0 to n) in metrics calculations

# https://www.cawcr.gov.au/projects/verification/

####--------------------------------------------------------------------------.
#### - Weights options

####--------------------------------------------------------------------------.
#### Skill scores
# - reference ...
# - persistence, climatology, ...

####--------------------------------------------------------------------------.
#### - pandas:
# - example to_xarray for exploting xverif ...
# - exploit stacked code ...

####--------------------------------------------------------------------------.
#### Dataset preprocessing
# Preprocessing on stacked 2D array (per chunk within ufunc) or native Dataset ?
# If loop over 1D, drop nan. If vectorize, need to use nanfunctions


# --> Drop nans
# --> Drop inf
# Drop or masking operations for continuous verification
# - Drop pairwise_equal_elements (i.e. 0)
# - Keep only within a value range  (single conditions (or), double (and))
#   --> Add option for multiple ranges ? >0.5, >30, >60
#   --> If yes, means Dataset size changes
# --> Dropping cause array size to change
# --> Masking (with np.nan), not dropping nan and metric dealing with nan?


# conditioning: {None, "single", "double"}, optional
# The type of conditioning used for the verification.
# The default, conditioning=None, includes all pairs. With
# conditioning="single", only pairs with either pred or obs > thr are
# included. With conditioning="double", only pairs with both pred and
# obs > thr are included.

####--------------------------------------------------------------------------.
#### Continuous verification
# TODO robust with median and IQR / MAD
# - scatter
# - smape: https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L802
# - mape: https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/deterministic.py#L1215
# - add n_obs

# epsilon is an arbitrary small yet strictly positive number to avoid undefined results when ``a`` is zero.

#### xskillscore
#   Mean Absolute Percentage Error
# -   Percent error is reported as decimal percent (1-100)
# epsilon = np.finfo(np.float64).eps
# mape = np.abs(error) / np.maximum(np.absolute(obs), epsilon)

# # Symmetric Mean Absolute Percentage Error
# smape = np.absolute(error) / (np.absolute(obs) + np.absolute(pred))

# # Median Absolute Error
# rob_MAE = np.median(np.absolute(error))

#### pysteps
# NMSE normalized mean squared error
# E[(pred - obs)^2]/E[(pred + obs)^2].
# --> NMSE = MSE / (pred + obs) ** 2

# DRMSE debiased root mean squared error
# DRMSE = np.sqrt(MSE - BIAS**2)


# beta (slope linear regression model)
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L162

# beta1 (degree of conditional bias of the observations given the forecasts (type 1)
# beta2  (degree of conditional bias of the forecasts given the observation (type 2)

# robust option --> mean --> median , std --> mad

# diffMedian
# rMedian

# diffMAD  (median absolute deviation)
# rMAD

# rob_obs_CoV (MAD/median)
# rob_pred_CoV (MAD/median)
# rob_diffCoV

# relBIAS (BIAS/obs_mean)

# rob_BIAS (median(error)
# rob_relBIAS:  robBIAS/obs_median

# relMAE  (MAE/obs_mean)
# rob_MAE  (median(abs(error)))
# rob_relMAE  robMAE/obs_median

# relMSE
# rob_MSE
# rob_relMSE


# err_AC1 k=1
# err_AC2 k=2: error autocorrelation at lag ...

####--------------------------------------------------------------------------.
### Categorical
# - Gerriity skill Score
# --> https://rdrr.io/github/joaofgoncalves/SegOptim/man/GerritySkillScore.html
# --> https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/contingency.py#L735
# --> https://www.ecmwf.int/sites/default/files/elibrary/2010/11988-new-equitable-score-suitable-verifying-precipitation-nwp.pdf

# Add np.log10 everywhere !!!
# - In categorical skills
