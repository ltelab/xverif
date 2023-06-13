#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:45:52 2023.

@author: ghiggi
"""
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from xverif.preprocessing import _drop_infs, _drop_nans, _drop_pairwise_elements
from xverif.utils.timing import print_elapsed_time

# -----------------------------------------------------------------------------.
# #########################################
#### Deterministic categorical metrics ####
# #########################################


def _det_cat_metrics(
    pred, obs, thr=0.000001, skip_na=True, skip_infs=True, skip_zeros=True
):
    """Compute deterministic metrics for categorical binary predictions.

    This function expects pred and obs to be 1D vector of same size.
    """
    # TODO robust with median and IQR / MAD
    ##------------------------------------------------------------------------.
    # Preprocess data (remove NaN if asked)
    if skip_na:
        pred, obs, _ = _drop_nans(pred, obs)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 2:
            return np.ones(12) * np.nan

    # Preprocess data (remove NaN if asked)
    if skip_infs:
        pred, obs, _ = _drop_infs(pred, obs)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 2:
            return np.ones(12) * np.nan

    if skip_zeros:
        pred, obs, _ = _drop_pairwise_elements(pred, obs, element=0)
        # If not non-NaN data, return a vector of nan data
        if len(pred) < 1:
            return np.ones(12) * np.nan

    # apply threshold
    predb = pred > thr
    obsb = obs > thr

    # calculate hits, misses, false positives, correct rejects
    H = np.nansum(np.logical_and(predb == 1, obsb == 1), dtype="float64")
    F = np.nansum(np.logical_and(predb == 1, obsb == 0), dtype="float64")
    M = np.nansum(np.logical_and(predb == 0, obsb == 1), dtype="float64")
    R = np.nansum(np.logical_and(predb == 0, obsb == 0), dtype="float64")

    # Probability of detection
    POD = H / (H + M)
    # False alarm ratio
    FAR = F / (H + F)
    # False alaram rate (prob of false detection)
    FA = F / (F + R)
    s = (H + M) / (H + M + F + R)

    # Accuracy (fraction correct)
    ACC = (H + R) / (H + M + F + R)
    # Critical success index
    CSI = H / (H + M + F)
    # Frequency bias
    FB = (H + F) / (H + M)

    # Heidke Skill Score (-1 < HSS < 1) < 0 implies no skill
    HSS = 2 * (H * R - F * M) / ((H + M) * (M + R) + (H + F) * (F + R))
    # Hanssen-Kuipers Discriminant
    HK = POD - FA

    # Gilbert Skill Score
    GSS = (POD - FA) / ((1 - s * POD) / (1 - s) + FA * (1 - s) / s)
    # Symmetric extremal dependence index
    SEDI = (np.log(FA) - np.log(POD) + np.log(1 - POD) - np.log(1 - FA)) / (
        np.log(FA) + np.log(POD) + np.log(1 - POD) + np.log(1 - FA)
    )
    # Matthews correlation coefficient
    MCC = (H * R - F * M) / np.sqrt((H + F) * (H + M) * (R + F) * (R + M))
    # F1 score
    F1 = 2 * H / (2 * H + F + M)

    skills = np.array([POD, FAR, FA, ACC, CSI, FB, HSS, HK, GSS, SEDI, MCC, F1])

    return skills


##----------------------------------------------------------------------------.


@print_elapsed_time(task="deterministic categorical")
def _deterministic_categorical_metrics(
    pred, obs, dim="time", skip_na=True, skip_infs=True, skip_zeros=True, thr=0.000001
):
    ds_skill = xr.apply_ufunc(
        _det_cat_metrics,
        pred,
        obs,
        kwargs={
            "thr": thr,
            "skip_na": skip_na,
            "skip_infs": skip_infs,
            "skip_zeros": skip_zeros,
        },
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["skill"]],  # returned data has one dimension
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {
                "skill": 12,
            }
        },
        output_dtypes=["float64"],
    )  # dtype

    # Compute the skills
    with ProgressBar():
        ds_skill = ds_skill.compute()

    # Add skill coordinates
    skill_str = [
        "POD",
        "FAR",
        "FA",
        "ACC",
        "CSI",
        "FB",
        "HSS",
        "HK",
        "GSS",
        "SEDI",
        "MCC",
        "F1",
    ]
    ds_skill = ds_skill.assign_coords({"skill": skill_str})
    ##------------------------------------------------------------------------.
    # Return the skill Dataset
    return ds_skill

