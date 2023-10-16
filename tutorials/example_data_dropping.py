#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:47:24 2023.

@author: ghiggi
"""
import numpy as np
from xverif.dropping import DropData

#### Example
### Numpy-based
pred = np.array([1.0, 1.0, 2.0, np.nan, 4.0, np.inf])
obs = np.array([1.0, 0.5, 2.2, np.nan, 3.8, np.inf])

# No dropping
pred, obs = DropData(pred, obs).apply()

# One-liner dropper
drop_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    {"equal_values": False},
    {"values": 0, "conditioned_on": "both"},
    {"below_threshold": 1.5, "conditioned_on": "both"},
    {"above_threshold": 3, "conditioned_on": "obs"},
]
pred, obs = DropData(pred, obs, drop_options=drop_options).apply()

pred, obs = DropData(pred, obs, drop_options={"nan": True}).apply()

# Step by step masking
pred, obs = DropData(pred, obs).nan()
pred, obs = DropData(pred, obs).inf()
pred, obs = DropData(pred, obs).values(values=0)
pred, obs = DropData(pred, obs).equal_values()
pred, obs = DropData(pred, obs).above_threshold(threshold=3)
pred, obs = DropData(pred, obs).below_threshold(threshold=3, conditioned_on="any")

# Check condition drop all
drop_options = [
    {"below_threshold": 1.5, "conditioned_on": "both"},
    {"above_threshold": 1.5, "conditioned_on": "obs"},
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    {"equal_values": False},
    {"values": 0, "conditioned_on": "both"},
]
pred, obs = DropData(pred, obs, drop_options=drop_options).apply()

# ------------------------------------------------------------------------------.
# TODO:  NOT IMPLEMENTED

# Create Xarray Dataset
# pred_data = xr.DataArray(
#     pred_dask, dims="dim", coords={"dim": range(pred_dask.shape[0])}
# )
# obs_data = xr.DataArray(obs_dask, dims="dim", coords={"dim": range(obs_dask.shape[0])})
# ds_pred = xr.Dataset({"var1": pred_data, "var2": pred_data})
# ds_obs = xr.Dataset({"var1": obs_data, "var2": obs_data})

# # Dataset masking
# drop_options = [
#     {"nan": True},
#     {"inf": True, "conditioned_on": "any"},
#     {"equal_values": False},
#     {"values": 0, "conditioned_on": "both"},
#     {"below_threshold": 1.5, "conditioned_on": "both"},
#     {"above_threshold": 3, "conditioned_on": "obs"},
# ]

# drop_options = {"var1": drop_options, "var2": [{"nan": True}]}
# ds_pred, ds_obs = mask_datasets(ds_pred, ds_obs, drop_options)
