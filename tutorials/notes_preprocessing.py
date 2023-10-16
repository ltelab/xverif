#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:24:32 2023.

@author: ghiggi
"""
#### Loop vs Vectorized Implementations
# Loop implementation
# --> Does not expect np.nan, np.inf
# --> Discard np.nan, np.inf, pairwise equal values, ... (if asked)
# --> NaN dropping INSIDE the metric computation routine

# Vectorized implementation
# --> Does expect np.nan
# --> Does not expect np.inf
# --> Set np.inf,pairwise equal values, ... to np.nan (if asked)
# --> NaN masking OUTSIDE the metric computation routine
# --> Return NaN if some NaN NOT IMPLEMENTED yet !
#     --> BUT output metrics can be masked afterwards

# Computational aspects
# - If Dataset contains lot of np.nan and data that should be discarded/masked,
#   implementation="loop" might be faster.
# - Verification considering multiple below/above thresholds
#   - either requires calling the xverif.<function> multiple times.

# TODO: either call XXXX to add thresholds dimension to Dataset before calling xverif !
# --> Useful for continuous, probability data_type !
# --> Similar for multiclass option: one vs. others !

## Technical points
# Vectorized implementation
# - Mask data !
# - Masking is done by chunk !
# - Chunks between pred and obs must be aligned
# - Expects dask arrays

# Loop implementation
# - Drop data !
# - Expect pred and obs 1D numpy vectors

# pandas & xarray functions only provide the skipna argument

####--------------------------------------------------------------------------.
#### Skip options (Masking & Dropping)
#### skip_options
# --> List
# - nan: True
# - inf: True
# - equal_values: False
# - values : [ ] (int, float, list)
# - above_threshold
# - below_threshold
# --> [], {} --> Do nothing

# Conditioning option
# - conditioned_on: 'both', 'any', 'pred', 'obs',

# Possible cases:
# - Various options and conditions
#   --> Mask values=[0] conditioned_on 'both',
#   --> Mask 'below_threshold' conditioned_on on 'obs',
#   --> Mask infinite conditioned_on 'any'

# - Mask outside a interval
#   - Specify above_threshold and below_threshold

# - Mask inside an interval
#   --> TODO: NOT IMPLEMENTED

# Dataset case
# - Different options per variable !
#     --> {var1: {}, var2: {}}
#     --> var not specified set to default masking options

### Vectorized thresholds
# - Masking above/below multiple thresholds
#   --> Add new dimension corresponding to the masking condition !

####--------------------------------------------------------------------------.
# Resampling / Bootstrapping:
# - https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/resampling.py
