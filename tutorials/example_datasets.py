#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:53:54 2023.

@author: ghiggi
"""
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
create_timeseries_dataset(1000, data_type="continuous")  # default
create_timeseries_dataset(1000, data_type="categorical", n_class=3)
create_timeseries_dataset(1000, data_type="probability", n_class=3)
create_timeseries_forecast_dataset(1000)

create_spatial2d_dataset(obs_number=1000)
create_spatial3d_dataset(obs_number=1000)
create_ensemble_dataset(obs_number=1000)
create_ensemble_forecast_dataset(obs_number=1000)
create_multimodel_dataset(obs_number=1000)
create_multimodel_ensemble_forecast_dataset(obs_number=1000)


create_spatial2d_dataset(
    obs_number=1000,
    obs_dim="time",
    data_type="continuous",
    aux_shape=(10, 10),
    aux_dims=["x", "y"],
    n_class=2,
    n_vars=3,
)
