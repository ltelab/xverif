#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:53:54 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from xverif.datasets import (
    create_ensemble_dataset,
    create_ensemble_forecast_dataset,
    create_multimodel_dataset,
    create_multimodel_ensemble_forecast_dataset,
    create_spatial2d_dataset,
    create_spatial3d_dataset,
    create_timeseries_dataset,
    create_timeseries_forecast_dataset,
)

# Simulate datasets

create_multimodel_ensemble_forecast_dataset(obs_number=1000, data_type="continuous")
create_multimodel_ensemble_forecast_dataset(obs_number=1000, data_type="categorical", n_class=3)
create_multimodel_dataset(obs_number=1000)
create_ensemble_forecast_dataset(obs_number=1000)
create_spatial2d_dataset(obs_number=1000)
create_spatial3d_dataset(obs_number=1000)
create_timeseries_dataset(1000)
create_timeseries_forecast_dataset(1000)

create_spatial2d_dataset(obs_number=1000,
                         obs_dim="time",
                         data_type="continuous",
                         aux_shape=(10,10),
                         aux_dims=["x", "y"],
                         n_class=2,
                         n_vars=3)

# Simulate obs and pred
obs = create_spatial2d_dataset(obs_number=1000)

pred = create_ensemble_dataset(obs_number=1000)
pred = create_ensemble_forecast_dataset(obs_number=1000)


# Broadcasting obs to pred
pred1, obs1 = xr.broadcast(pred, obs)
np.testing.assert_allclose(obs1.isel({"realization": 0})["var1"].data, obs1.isel({"realization":1})["var1"].data)


# Compute deterministic skills
from xverif.wrappers import deterministic

ds_skills = deterministic(
    pred=pred,
    obs=obs,
    forecast_type="continuous",
    aggregating_dim=["time"],
    skip_na=True,
    # TODO ADD
    skip_infs=True,
    skip_zeros=True,
    win_size=5,
    thr=0.000001,
)



