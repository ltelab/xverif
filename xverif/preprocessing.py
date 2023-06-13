#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:39:34 2023.

@author: ghiggi
"""
import numpy as np


def _match_nans(a, b, weights=None):
    """
    Considers missing values pairwise.

    If a value is missing in a, the corresponding value in b is turned to nan, and
    vice versa.

    Code source: https://github.com/xarray-contrib/xskillscore

    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with nans placed at pairwise locations.
    """
    if np.isnan(a).any() or np.isnan(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isnan(a), np.isnan(b))
        a[idx], b[idx] = np.nan, np.nan
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights[idx] = np.nan
    return a, b, weights


def _drop_nans(a, b, weights=None):
    """
    Considers missing values pairwise.

    If a value is missing in a or b, the corresponding indices are dropped.

    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with nans placed at pairwise locations.
    """
    if np.isnan(a).any() or np.isnan(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isnan(a), np.isnan(b))
        a = np.delete(a, idx)
        b = np.delete(b, idx)
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights = np.delete(weights, idx)
    return a, b, weights


def _drop_infs(a, b, weights=None):
    """
    Considers infinite values pairwise.

    If a value is infinite in a or b, the corresponding indices are dropped.

    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with infs placed at pairwise locations.
    """
    if np.isinf(a).any() or np.isinf(b).any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isinf(a), np.isinf(b))
        a = np.delete(a, idx)
        b = np.delete(b, idx)
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights = np.delete(weights, idx)
    return a, b, weights


def _drop_pairwise_elements(a, b, weights=None, element=0):
    """
    Considers values pairwise.

    If the element is the same in a and b, the corresponding indices are dropped.

    Returns
    -------
    a, b, weights : ndarray
        a, b, and weights (if not None) with elements placed at pairwise locations.
    """
    idx = np.logical_and(a == element, b == element)
    if idx.any():
        # Avoids mutating original arrays and bypasses read-only issue.
        a, b = a.copy(), b.copy()
        # Find pairwise indices in a and b that have nans.
        idx = np.logical_or(np.isinf(a), np.isinf(b))
        a = np.delete(a, idx)
        b = np.delete(b, idx)
        if isinstance(weights, np.ndarray):
            if weights.shape:  # not None
                weights = weights.copy()
                weights = np.delete(weights, idx)
    return a, b, weights


