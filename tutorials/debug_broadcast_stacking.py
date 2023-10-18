#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:30:27 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr
from xverif.datasets import (
    create_ensemble_dataset,
    create_ensemble_forecast_dataset,
    # create_multimodel_dataset,
    # create_multimodel_ensemble_forecast_dataset,
    create_spatial2d_dataset,
    # create_spatial3d_dataset,
    create_timeseries_dataset,
    create_timeseries_forecast_dataset,
)
from xverif.wrappers import align_xarray_objects, ensure_dataarray
from xverif.metrics.deterministic.continuous_vectorized import get_stacking_dict

##-----------------------------------------------------------------------------.
# Simulate obs and pred
obs = create_spatial2d_dataset(obs_number=1000)

pred = create_ensemble_dataset(obs_number=1000)
pred = create_ensemble_forecast_dataset(obs_number=1000)


# Broadcasting obs to pred
pred1, obs1 = xr.broadcast(pred, obs)
np.testing.assert_allclose(
    obs1.isel({"realization": 0})["var1"].data,
    obs1.isel({"realization": 1})["var1"].data,
)

# Align datasets
pred2, obs2 = align_xarray_objects(pred, obs)
##-----------------------------------------------------------------------------.

data_type = "continuous"
n_categories = None

data_type = "binary"
n_categories = 2

data_type = "multiclass"
n_categories = 4


obs = create_timeseries_dataset(100, data_type=data_type, n_categories=n_categories)
pred = create_timeseries_forecast_dataset(100, data_type=data_type, n_categories=n_categories)

pred['var0'].data


sample_dims = "time"
skip_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
]

###----------------------------------------------------------------------------
# Vectorized 
pred, obs = align_xarray_objects(pred, obs)
obs = obs.broadcast_like(pred)
pred = ensure_dataarray(pred)
obs = ensure_dataarray(obs)

###----------------------------------------------------------------------------
#  Vectorized Routine 
stacking_dict = get_stacking_dict(pred, sample_dims=sample_dims)
pred = pred.stack(stacking_dict)
obs = obs.stack(stacking_dict)


###----------------------------------------------------------------------------
# DEBUG 
obs_broadcasted = obs.broadcast_like(pred)
stacking_dict = get_stacking_dict(pred, sample_dims=sample_dims)
pred = pred.stack(stacking_dict)
obs = obs_broadcasted.stack(stacking_dict)


pred = pred.data
obs = obs.data