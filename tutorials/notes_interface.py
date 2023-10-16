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
# metric_type: deterministic, probabilistic (spatial, temporal)
# support_type: point, spatial, temporal

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
    aggregating_dim=["x", "y"],  # TODO: rename
    # Optional argument
    support_type_kwargs={},
    metrics=None,
    # Preprocessing
    skip_options=None,
    # Backend options
    implementation=None,  # TODO: support_type="point" --> vectorized
)

# ds_skills = xverif.probabilistic()

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
#### TODOs
# TODO (spatial, temporal)
# --> Can be deterministic or probabilistic metrics
# --> Use of support_type ?

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
