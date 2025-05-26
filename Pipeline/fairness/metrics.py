import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from scipy.stats import ks_2samp


# -------------------------------------------------------------
# helpers
# -------------------------------------------------------------
def _split_users_by_gender(ml_user_df):
    """Return two arrays of user-ids: males, females"""
    males   = ml_user_df.loc[ml_user_df.gender == 'M', 'user_id'].values
    females = ml_user_df.loc[ml_user_df.gender == 'F', 'user_id'].values
    return males, females

def _item_pop_bins(train_inter_df, n_bins=2):
    """Return dict:  item_id -> bin (0 … n_bins-1)  by popularity"""
    pop = Counter(train_inter_df.item_id)
    items, counts = zip(*pop.items())
    quantiles = np.quantile(counts, np.linspace(0,1,n_bins+1))
    return {it: int(np.searchsorted(quantiles, c, side='right')-1) for it,c in pop.items()}

# -------------------------------------------------------------
# fairness metrics  that work on ML-1M
# -------------------------------------------------------------
def absolute_difference(group1_scores, group2_scores):
    return abs(group1_scores.mean() - group2_scores.mean())

def gini(array):
    """Gini coefficient over positive values"""
    array = np.array(array, dtype=np.float64) + 1e-9
    array = np.sort(array)
    n = array.size
    coef = (2*np.arange(1,n+1)-n-1).dot(array) / (n*array.sum())
    return coef

# def variance(array):
#     # return np.var(array)
#     # TODO: ddof=1 or 0?
#     return np.var(array, ddof=1)

def variance(array):
    arr = np.asarray(array, dtype=float)
    n = arr.size

    diffs = np.subtract.outer(arr, arr)
    return ((diffs**2).sum()) / (n**2)


# -------------------------------------------
#  Group-level error-gap helpers
# -------------------------------------------
def _per_item_group_errors(df, group_col, err_col):
    """
    df: dataframe with columns [item_id, group_col, err_col]
    returns: dict item_id -> {group:value}
    """
    d = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        d[row.item_id][row[group_col]].append(row[err_col])
    # aggregate (mean) per item & group
    agg = {it: {g: np.mean(vs) for g,vs in groups.items()} for it,groups in d.items()}
    return agg

# ------------ BASIC METRICS ----------------
def mae_gap(err_g1, err_g2):
    """absolute gap between two groups' MAE"""
    return abs(err_g1.mean() - err_g2.mean())

# ------------ UNFAIRNESS family ------------

def value_unfairness(item_group_err):
    """U_val = mean | (E0[err] − E1[err]) |"""
    diffs = [
        abs(values.get(1,0) - values.get(2,0))
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

def absolute_unfairness(item_group_err):
    """U_abs = mean | |E0[err]| − |E1[err]| |"""
    diffs = [
        abs(abs(values.get(1,0)) - abs(values.get(2,0)))
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

def over_unfairness(item_group_err):
    """U_under = mean | max(0, (E0[err_real]-E0[err_pred])) − max(0, E1[err_real]-E1[error_pred]) |"""

    # multiply by -1 in order to flip from (real-pred) to (pred-real) (to match definition)
    diffs = [
        abs(
            max((values.get(1,0)*-1),0) -
            max((values.get(2,0)*-1),0)
        )
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

def under_unfairness(item_group_err):
    diffs = [
        abs(
            max(values.get(1,0),0) -
            max(values.get(2,0),0)
        )
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

# ------------- OTHER ------------------------
def min_max_diff(array):
    return np.max(array) - np.min(array)

def ks_statistic(sample1, sample2):
    return ks_2samp(sample1, sample2).statistic

import numpy as np

def ks_area_difference(sample1, sample2, n_bins=100):
    """
    Compute the L1 “area” between two empirical CDFs:
      KS_area = sum_{i=1}^T l * |F1(i) - F2(i)|

    sample1, sample2 : 1D arrays of your utilities
    n_bins           : number of equal-width intervals T

    Returns the total area difference.
    """
    # 1) define common bin edges over the full range
    mn = min(sample1.min(), sample2.min())
    mx = max(sample1.max(), sample2.max())
    bins = np.linspace(mn, mx, n_bins + 1)

    # 2) histogram counts in each bin
    h1, _ = np.histogram(sample1, bins=bins)
    h2, _ = np.histogram(sample2, bins=bins)

    # 3) empirical CDF values at each bin edge (excluding the final edge)
    cdf1 = np.cumsum(h1) / sample1.size
    cdf2 = np.cumsum(h2) / sample2.size

    # 4) interval width l
    widths = np.diff(bins)  # length = n_bins

    # 5) area = sum over bins of (width * abs(cdf1 - cdf2))
    area = np.sum(widths * np.abs(cdf1 - cdf2))
    return area
