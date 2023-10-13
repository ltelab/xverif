#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:42:32 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr
import dask
from xverif.masking import MaskingDataArrays

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

masking_options = [
    {'nan': True},
    {'inf': True, 'conditioned_on': 'any'},
    {'equal_values': False},
    {'values': 0, 'conditioned_on': 'both'},
    {'below_threshold': 1.5, 'conditioned_on': 'both'},
    {'above_threshold': 3, 'conditioned_on': 'obs'},
]

masking = MaskingDataArrays(pred, obs, masking_options=masking_options)
masked_pred, masked_obs = masking.apply()

pred, obs = MaskingDataArrays(pred, obs, masking_options).apply()

obs.compute()
pred.compute()

pred, obs = MaskingDataArrays(pred, obs, masking_options={"nan": True}).apply()

masking_options={"nan": True}
