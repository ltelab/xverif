#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:08:07 2023.

@author: ghiggi
"""
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

# Spatial metrics
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/salscores.py
# - https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/spatialscores.py

# Hypothesis tests
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/stattests.py
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/comparative.py

# Effective sample size
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/np_deterministic.py#L109

####--------------------------------------------------------------------------.
#### Continuous verification
# TODO robust with median and IQR / MAD

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
#### Categorical binary
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


####--------------------------------------------------------------------------.
#### MultiClass
# - Collections of binary problems
# - Add dimension avg_method: [macro, micro, weighted]
# - https://permetrics.readthedocs.io/en/latest/pages/classification.html
# - https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification
# - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

# - https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics

# https://www.cawcr.gov.au/projects/verification/


####--------------------------------------------------------------------------.
#### - Weights options


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
