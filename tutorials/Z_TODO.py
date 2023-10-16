#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:26:23 2023.

@author: ghiggi
"""
####--------------------------------------------------------------------------.
##############
#### TODO ####
##############
# Define better naming dimensions
# sample_dims --> sample_dims
# - sample_dims --> None?
# - Change name of sample_dims
# - Change name of sample and aux dimensions

# Rename forecast_type --> data_type


# Return same skills between loop and vectorized

# Working loop and vectorized solutions
# - Deterministic Continuous and Binary

# Expand thresholds (category ...) --> Add new dimension

# Categorize / Binning (xcategorize, xbinning)¨

# Support dimensions
# --> Should also be included in sample_dims ?
# --> Spatial metrics (stacked_dims, x, y, sample_dims)

####--------------------------------------------------------------------------.
#### Daskify functions

####--------------------------------------------------------------------------.
#### - Suppress warnings
# FutureWarning: None value for "chunks" is deprecated. It will raise an error in the future. Use instead "{}"
# RuntimeWarning: invalid value encountered in divides
# RuntimeWarning: All-NaN slice encountered

# "invalid value encountered in divide"  # division by 0
# "Degrees of freedom <= 0 for slice"  # (when all np.nan in nanstd)
# "All-NaN slice encountered"
# "invalid value encountered in double_scalars"

# sklearn way to deal with division by 0
# https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/metrics/_classification.py#L1301

####--------------------------------------------------------------------------.
#### Check n. tasks
# - xverif.suppress_warnings
# - Check n tasks
#   - too not be too high  (use max threshold ..)
# - Check data chunks
#   - too not be too large (check memory available)
#   - too not be too small (compared to dataset size)

# --> Suggest sensible rechunking !

####--------------------------------------------------------------------------.
#### Metrics info
# - names.py
#   - standard_name, long_name, description, units (CF)
#   - pycolorbar settings (cmap, vmin, vmax, extend, title)

####--------------------------------------------------------------------------.
