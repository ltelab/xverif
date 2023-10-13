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
# - https://permetrics.readthedocs.io/en/latest/pages/regression.html

# Deterministic categorical
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/contingency.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/detcatscores.py
# --> Tests: https://github.com/pySTEPS/pysteps/blob/master/pysteps/tests/test_verification_detcatscores.py
# - https://permetrics.readthedocs.io/en/latest/pages/classification.html
# - https://www.debadityachakravorty.com/ai-ml/cmatrix/

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
#   --> Preprocessing apply to each pixel timeseries separately
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
"invalid value encountered in divide"  # division by 0
"Degrees of freedom <= 0 for slice"  # (when all np.nan in nanstd)
"All-NaN slice encountered"
"invalid value encountered in double_scalars"

# sklearn way to deal with division by 0
# https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/metrics/_classification.py#L1301

####--------------------------------------------------------------------------.
#### - Categorical verification
# --> Binary or multiclass (separated)
# --> If probababilities, add tool to predict classes using various probabilities (--> prob_threshold dimension)
# --> Expect classes (0,1) or (0 to n) in metrics calculations

# https://www.cawcr.gov.au/projects/verification/

####--------------------------------------------------------------------------.
#### - Weights options

####--------------------------------------------------------------------------.
#### - pandas:
# - example to_xarray for exploiting xverif ...
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
#### Interface
# xverif.metric("MSE", pred, obs, ...)
# - YAML file with all available metrics
# - Path to module, file


####--------------------------------------------------------------------------.
#### Continuous verification
# - MAPE == relMAE  (can exceed 100 % !)
# - Do include SMAPE ? (bounded between 0 and 200)
# --> https://typethepipe.com/post/symmetric-mape-is-not-symmetric/
# --> https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac
# --> https://medium.com/@davide.sarra/how-to-interpret-smape-just-like-mape-bf799ba03bdc
# --> https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
# --> Confusing:
#     https://permetrics.readthedocs.io/en/latest/pages/regression/MPE.html
#     https://permetrics.readthedocs.io/en/latest/pages/regression/MRE.html
#     https://permetrics.readthedocs.io/en/latest/pages/regression/MRE.html

# NMSE normalized mean squared error (pysteps)
# E[(pred - obs)^2]/E[(pred + obs)^2].
# --> NMSE = MSE / (pred + obs) ** 2

# TotalDeviationRatio
# TotalDeviationRatio = obs.nansum(axis=1)/pred.nansum(axis=1)

# # SMAPE
# error_abs_perc_sym = np.absolute(error) / (np.absolute(obs) + np.absolute(pred)) / 2
# SMAPE = error_abs_perc_sym.nanmean(axis=1) # between 0 and 200

# # AAPE
# error_arctan_abs_perc = np.arctan(error_abs_perc)

# # MAAPE
# # - Mean arctangent absolute percentage error
# # - https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error
# MAAPE = error_arctan_abs_perc.nanmean(axis=1)

# # MSLE
# # - MeanSquaredLogarithmic Error
# # - Natural log ?
# error_log = np.log(1 + obs) - np.log(1 + pred)
# error_log = np.log((1 + pred)/ (1 + obs))
# MSLE = np.nanmean(error_log**2)


# # A10, A20, A30 index
# # percentage of predictions with absolute percentage error < 0.1, 0.2 or 0.3

# # Percentage better ...


# accuracy_ratio = pred/obs
# error_log = np.log(pred/obs)   # log accuracy ratio

# DRMSE debiased root mean squared error (pysteps)
# DRMSE = np.sqrt(MSE - BIAS**2)

# beta (slope linear regression model)
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L162

# beta1 (degree of conditional bias of the observations given the forecasts (type 1)
# beta2  (degree of conditional bias of the forecasts given the observation (type 2)

# residual standard error

# Distribution distance?
# KL, Jannon, Wasserstein?


####--------------------------------------------------------------------------.
### Categorical binary
# - Gerrity skill Score
# --> https://rdrr.io/github/joaofgoncalves/SegOptim/man/GerritySkillScore.html
# --> https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/contingency.py#L735
# --> https://www.ecmwf.int/sites/default/files/elibrary/2010/11988-new-equitable-score-suitable-verifying-precipitation-nwp.pdf

# Add np.log10 everywhere !!!
# - In categorical skills

# fbeta
# - weighted harmonic mean of precision and recall
# beta < 1 lends more weight to precision, while beta > 1 favors recall
# beta = 1 --> F1
# beta = 2 --> F2
# beta = 0 considers only precision
# beta = np.inf considers only recall
# - https://towardsdatascience.com/is-f1-the-appropriate-criterion-to-use-what-about-f2-f3-f-beta-4bd8ef17e285

# fbeta = (1 + beta**2) * (precision * POD) / (beta**2 * precision + POD)

# Hamming Score
# - fraction of labels that are incorrectly predicted.
# - zero_one_loss  = 1 - ACC
# HS = (F + M) / N

# Hamming Loss (HL)
# Hinge loss
# Zero_one_loss

# MultiClass
# - Collections of binary problems
# - Add dimension avg_method: [macro, micro, weighted]
# - https://permetrics.readthedocs.io/en/latest/pages/classification.html
# - https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
# - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

# - https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics

### Categorical binary (probability)
# - GINI
# - ROC-AUC score
# - logloss
# - brier score
# - Kolmogorov-Smirnov statistic
# --> https://neptune.ai/blog/evaluation-metrics-binary-classification


####--------------------------------------------------------------------------.
#### Temporal metrics
# err_AC1 k=1
# err_AC2 k=2: error autocorrelation at lag ...

# MDA = Mean directional accuracy
# - Check change of direction accuracy
# - Also called PCD (Prediction of Change in Direction)
# - https://en.wikipedia.org/wiki/Mean_directional_accuracy
# - https://support.numxl.com/hc/en-us/articles/360029220972-MDA-Mean-Directional-Accuracy

# MASE = Mean absolute scaled error
# --> https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
# MASE = MAE / MAE_lagged_prediction

# Anomaly correlation


# DTW distance metrics

####--------------------------------------------------------------------------.
#### Skills Scores
# - reference ...
# - persistence, climatology, ...


####--------------------------------------------------------------------------.
#### Spatial metrics


####--------------------------------------------------------------------------.
