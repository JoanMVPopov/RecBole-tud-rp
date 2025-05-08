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

def variance(array):
    return np.var(array)

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
    """U_val  (signed error difference)"""
    diffs = [
        abs(values.get(0,0) - values.get(1,0))
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

def absolute_unfairness(item_group_err):
    """U_abs  (absolute error magnitude difference)"""
    diffs = [
        abs(abs(values.get(0,0)) - abs(values.get(1,0)))
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

def under_unfairness(item_group_err):
    diffs = [
        abs(
            max(values.get(0,0)*-1,0) -
            max(values.get(1,0)*-1,0)
        )
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

def over_unfairness(item_group_err):
    diffs = [
        abs(
            max(values.get(0,0),0) -
            max(values.get(1,0),0)
        )
        for values in item_group_err.values()
    ]
    return np.mean(diffs)

# ------------- OTHER ------------------------
def min_max_diff(array):
    return np.max(array) - np.min(array)

def ks_statistic(sample1, sample2):
    return ks_2samp(sample1, sample2).statistic