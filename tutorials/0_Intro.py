#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:27:44 2023

@author: ghiggi
"""

# In the dynamic landscape of forecasting and prediction, verification remains a cornerstone. 
# To effectively evaluate the accuracy and reliability of models, a nuanced approach is essential.
#  Our software, xverif, has been meticulously designed to cater to this complexity, 
#  offering a systematic framework for verifying predictions across a myriad of applications.

# 'xverif' is engineered to handle n-dimensional data arrays using xarray, 
# ensuring optimized computations through vectorized operations and harnessing
#  the power of Dask for efficient parallelized computations on all cores of your machine(s).
 
# At the heart of xverif's verification system lie three foundational classifications:

# Data Type:

# Binary: Where predictions fall into one of two categories.
# Multiclass: Predictions belong to one of several discrete classes.
# Ordered (Ordinal): Predictions are categorized into a hierarchy (e.g., low, medium, high). Metrics such as weighted kappa or the cumulative log odds ratio can be employed to gauge accuracy.
# Probability: Predictions provide a probability distribution across possible outcomes.
# Continuous: Predictions yield real-valued outcomes.
# Count: Predictions result in non-negative integer values. For these, we can use metrics like Poisson or Negative Binomial log likelihood, and specialized metrics like the Mean Absolute Scaled Error (MASE) for time series count data.

# Support Type:

# Point: Treats each observation and prediction as an independent entity.
# Spatial: Captures the spatial interdependence of observations and predictions.
# Temporal: Highlights the temporal sequence and interrelation of predictions.
# Nature of Metrics:

# Metric Type
# Deterministic: Evaluate single models' prediction. Metrics such as Accuracy, Precision, Recall for binary data, or Mean Absolute Error for continuous data are part of this category.
# Probabilistic: Tailored for predictions that present a range of potential outcomes with associated probabilities. For instance, Log Loss for binary predictions or the Continuous Ranked Probability Score (CRPS) for continuous data.
# With this structured classification, xverif provides a granular verification system that can be tailored to specific needs, whether you're dealing with binary classifications in healthcare, spatially interdependent predictions in climate science, or probabilistic forecasts in finance.


# Through xverif, we aim to provide stakeholders from diverse domains the tools they need to ensure their predictions are not just accurate, but also meaningful and actionable.

# By offering this multifaceted tool, xverif facilitates a precise and adaptable verification system. 
# Whether you're assessing binary classifications in medical diagnostics, spatial predictions in geosciences, 
# , evaluating deterministic or probabilistic weather or financial forecasts, 
# xverif ensures your evaluations are both robust and relevant.