#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:24:32 2023

@author: ghiggi
"""
####--------------------------------------------------------------------------.
#### Dataset preprocessing
# Preprocessing on stacked 2D array (per chunk within ufunc) or native Dataset ?
# If loop over 1D, drop nan. If vectorize, need to use nanfunctions


# --> Drop nans
# --> Drop inf
# Drop or masking operations for continuous verification
# - Drop pairwise_equal_elements (i.e. 0)
# - Keep only within a value range  (single conditions (or), double (and))
#   --> Add option for multiple ranges ? >0.5, >30, >60
#   --> If yes, means Dataset size changes
# --> Dropping cause array size to change
# --> Masking (with np.nan), not dropping nan and metric dealing with nan?


# conditioning: {None, "single", "double"}, optional
# The type of conditioning used for the verification.
# The default, conditioning=None, includes all pairs. With
# conditioning="single", only pairs with either pred or obs > thr are
# included. With conditioning="double", only pairs with both pred and
# obs > thr are included.

# Preprocessing
# https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/utils.py


####--------------------------------------------------------------------------.
#### Chunk preprocessing
# - Point/Pixel-wise
#   --> Preprocessing apply to each samples vector (i.e. pixel timeseries) separately
#   --> ds_forecast.chunk({"time": -1}

# - Spatial chunks
#   --> Preprocessing apply to spatial region
#   --> If given timestep only one pixel nan (in obs or pred), all timesteps becomes nan
#   --> Or metrics deals with np.nan (but not efficient?)
#   --> Or looping over dimensions with numba and avoid vectorization?

####--------------------------------------------------------------------------.
# Resampling / Bootstrapping:
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/resampling.py