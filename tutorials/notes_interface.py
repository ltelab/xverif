#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:22:02 2023.

@author: ghiggi
"""
# https://github.com/xarray-contrib/xskillscore/tree/main/xskillscore/core
# https://github.com/pySTEPS/pysteps/tree/master/pysteps/verification
# https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/interface.py

####--------------------------------------------------------------------------.
#### Data, Metric, Support Type
# data_type: binary, multiclass, ordered, probability, continuous
# metric_type: deterministic, probabilistic
# support_type: point, spatial, temporal

####--------------------------------------------------------------------------.
#### Dimension names
#### Sample Dimensions (sample_dims)
# - Dimensions to be considered as sample observations
# - Dimension over which reduction/aggregation is performed
# - Dimensions that must be available in both prediction and observation datasets.
# - Dimensions over which the data should be unchunked (or coarsely chunked)
# - Sample dimensions are flattened to a single dimension before computing the metrics
# - For deterministic point metrics, the remaining dimensions are stacked into another single dimension ('aux')
# - For probabilistic point metrics, the remaining dimensions are stacked into the 'aux' and 'realizations' dimensions.

# - Example: pixelwise skills: sample_dims='time'           -> skills = f(x,y)
# - Example: metrics at each timestep: sample_dims='x','y'  -> skills = f(time)
# - For performance reasons, rechunk sample dimensions to -1 (unique chunking)
#   --> ds_pred.chunk({"<sample_dim>": -1}

#### Broadcast dimensions (broacast_dims)
# - Dimensions present in only 1 dataset (usually the prediction dataset)
# - Broadcasting (of observation to predictions) enable to have same dimensions
# - Typical broadcasting dimensions: forecast_leadtime, realizations

#### Auxiliary dimensions (aux_dims)
# - Dimensions which are not sample dimensions and not support dimensions
# - Skills will be computed at each point of these dimensions

#### Support dimensions
#### - Spatial/Image dimensions (spatial_dims)
# - For spatial metrics only

#### - Temporal dimension  (time_dim)
# - For temporal metrics only

#### - Realizations dimension (realizations_dim)
# - For probabilistic metrics only


# Point metrics are applied over flattened sample_dims + support_dimensions
# - Vectorized implementation applies over n-dimensional arrays
#   --> Deterministic: row: stacked dimensions, column: flattened sample dimensions
#   --> Probabilistic: row: stacked dimensions, column: flattened sample dimensions, 3rd: realizations

# - Loop implementation applies over the sample vector separately
#   --> Deterministic: flattened sample dimensions,
#   --> Probabilistic: row: flattened sample dimensions, column: realizations

####--------------------------------------------------------------------------.
#### Interface
import xarray as xr
import xverif

metric_type = None
data_type = None
xverif.available_metrics(metric_type, data_type)

xverif.deterministic.available_metrics(data_type)
xverif.probabilistic.available_metrics(data_type)

xverif.deterministic.available_data_type()
xverif.deterministic.available_support_type()
xverif.probabilistic.available_data_type()
xverif.deterministic.available_support_type()

xverif.available_metric_type()

pred = xr.Dataset()
obs = xr.Dataset()

ds_skills = xverif.deterministic(
    pred=pred,
    obs=obs,
    # Data and metric information
    data_type="continuous",
    support_type="point",
    sample_dims=["x", "y"],
    # Optional argument
    support_type_kwargs={},
    metrics=None,
    # Preprocessing
    skip_options=None,
    # Backend options
    implementation=None,  # TODO: support_type="point"
)

# ds_skills = xverif.probabilistic()

####--------------------------------------------------------------------------.
#### TODOs
# TODO (spatial, temporal)
# --> Can be deterministic or probabilistic metrics
# --> Use of support_type ?
# --> Spatial metrics:
#  - applied to entire input spatial image
#  - applied on sliding window

# Temporal metrics
# - Require time index
# - Time scale of interest ?
# - Daily/Weekly/Monthly/Seasonal/Annual ...


# TODO Distribution metrics
# --> in which <XXXX_type>?

# TODO DESIGN
# --> How to apply to single sample vectors --> xverif.compute_metric("MSE", pred, obs)
# --> How to apply single metric function --> xverif.metric("MSE)
# --> Pandas interface: to_xarray for exploiting xverif ?

# TODO:
# - Statistical tests
# - Resampling / Bootstrapping
# - Distribution metrics?


# Others possibilities
# - YAML file with all available metrics
#   - Path to module, file

####--------------------------------------------------------------------------.
