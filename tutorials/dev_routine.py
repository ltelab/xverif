#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 09:45:01 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr
import xverif
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
from xverif.wrappers import align_xarray_objects
from xverif.wrappers import align_xarray_objects, ensure_dataarray
from xverif.metrics.deterministic.continuous_vectorized import get_stacking_dict


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
# Loop 
pred, obs = align_xarray_objects(pred, obs)
pred = ensure_dataarray(pred)
obs = ensure_dataarray(obs)

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
# Multiclass 


 






    

