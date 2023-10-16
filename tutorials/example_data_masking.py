#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:42:32 2023.

@author: ghiggi
"""
import dask
import dask.array
import numpy as np
import xarray as xr
from xverif.masking import MaskDataArrays, mask_datasets

#### Example
### Numpy-based
pred_np = np.array([1.0, 1.0, 2.0, np.nan, 4.0, np.inf])
obs_np = np.array([1.0, 0.5, 2.2, np.nan, 3.8, np.inf])

pred = xr.DataArray(pred_np)
obs = xr.DataArray(obs_np)

# Dask-based
chunk_size = 2  # Adjust this value to your desired chunk size
pred_dask = dask.array.from_array(pred_np, chunks=chunk_size)
obs_dask = dask.array.from_array(obs_np, chunks=chunk_size)

# Create Xarray DataArrays
pred = xr.DataArray(pred_dask, dims="dim", coords={"dim": range(pred_dask.shape[0])})
obs = xr.DataArray(obs_dask, dims="dim", coords={"dim": range(obs_dask.shape[0])})

# No masking
masked_pred, masked_obs = MaskDataArrays(pred, obs).apply()

# One-liner masking
masking_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    {"equal_values": False},
    {"values": 0, "conditioned_on": "both"},
    {"below_threshold": 1.5, "conditioned_on": "both"},
    {"above_threshold": 3, "conditioned_on": "obs"},
]
masked_pred, masked_obs = MaskDataArrays(
    pred, obs, masking_options=masking_options
).apply()
masked_pred.compute()
masked_obs.compute()

pred, obs = MaskDataArrays(pred, obs, masking_options={"nan": True}).apply()

# Step by step masking
pred, obs = MaskDataArrays(pred, obs).nan()
pred, obs = MaskDataArrays(pred, obs).inf()
pred, obs = MaskDataArrays(pred, obs).values(values=0)
pred, obs = MaskDataArrays(pred, obs).equal_values()
pred, obs = MaskDataArrays(pred, obs).above_threshold(threshold=3)
pred, obs = MaskDataArrays(pred, obs).below_threshold(threshold=3, conditioned_on="any")


# Create Xarray Dataset
pred_data = xr.DataArray(
    pred_dask, dims="dim", coords={"dim": range(pred_dask.shape[0])}
)
obs_data = xr.DataArray(obs_dask, dims="dim", coords={"dim": range(obs_dask.shape[0])})
ds_pred = xr.Dataset({"var1": pred_data, "var2": pred_data})
ds_obs = xr.Dataset({"var1": obs_data, "var2": obs_data})

# Dataset masking
masking_options = [
    {"nan": True},
    {"inf": True, "conditioned_on": "any"},
    {"equal_values": False},
    {"values": 0, "conditioned_on": "both"},
    {"below_threshold": 1.5, "conditioned_on": "both"},
    {"above_threshold": 3, "conditioned_on": "obs"},
]

masking_options = {"var1": masking_options, "var2": [{"nan": True}]}
ds_pred, ds_obs = mask_datasets(ds_pred, ds_obs, masking_options)
