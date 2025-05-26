# -------------------------------------------------------------
#  Unified fairness evaluation for MovieLens-100K + RecBole model
# -------------------------------------------------------------
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
import json, torch, numpy as np, pandas as pd
from collections import defaultdict

import pandas
from tqdm import tqdm
import os

from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

# ---------- your own helpers -------------------------------
from fairness.metrics import (
    # group-error metrics
    _per_item_group_errors, mae_gap, value_unfairness,
    absolute_unfairness, under_unfairness, over_unfairness, ks_statistic,
    # utility / exposure metrics
    absolute_difference, variance, gini
)

# ---------- choose model / dataset --------------------------
MODEL    = 'CKE'        # CKE model as specified
DATASET  = 'ml-100k'    # RecBole's MovieLens-100K alias
TOPK     = 10

# ============================================================
#  A)  Load model & data  (+ full predictions, top-k lists)
# ============================================================
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        f'saved/CKE-May-14-2025_11-49-48.pth')

test_data2 = None
import pickle
with open("split_data.pth", 'rb') as f:
    test_data2 = pickle.load(f)

# Convert Interaction → pandas DataFrame
df1 = pd.DataFrame({k: v.cpu().numpy() for k, v in test_data.dataset.inter_feat.interaction.items()})
df2 = pd.DataFrame({k: v.cpu().numpy() for k, v in test_data2.dataset.inter_feat.interaction.items()})

# Sort for fair comparison
df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

model.eval()
device = model.device

# ---------- test interactions (point-wise) -------------------
# Use correct field names based on the config
uid_field = config['USER_ID_FIELD']
iid_field = config['ITEM_ID_FIELD']
rating_field = config['RATING_FIELD']


# Obtain data from the test_data loader
# We convert the interactions from the test data loader to tensors
u_t = test_data.dataset.inter_feat[uid_field].to(device)
i_t = test_data.dataset.inter_feat[iid_field].to(device)
r_t = test_data.dataset.inter_feat[rating_field].to(device)

intr = Interaction({uid_field: u_t, iid_field: i_t})
with torch.no_grad():
    preds = model.predict(intr).cpu().numpy()

err_signed = (r_t - preds).cpu().numpy()  # signed error

err_df = pd.DataFrame({
    uid_field: u_t.cpu().numpy(),
    iid_field: i_t.cpu().numpy(),
    'error': err_signed
})


# =================================================================
#  B)  Attach meta (gender, popularity)  +  compute fairness stats
# =================================================================
# ---------- Extract user and item metadata from dataset ----------

# 1) Access user features from RecBole dataset
if hasattr(dataset, 'user_feat') and dataset.user_feat is not None:
    # Create DataFrame from user features
    user_meta_data = {}
    user_meta_data[uid_field] = dataset.user_feat[uid_field].numpy()

    # Check if gender is available in user_feat
    if 'gender' in dataset.user_feat:
        user_meta_data['gender'] = dataset.user_feat['gender'].numpy()
    else:
        # If gender isn't available in user_feat, we need to handle it
        # For now, let's create a mock gender (50/50 split) for demonstration
        np.random.seed(42)  # For reproducibility
        user_meta_data['gender'] = np.random.choice(['M', 'F'], size=len(user_meta_data[uid_field]))
        print("WARNING: Gender data not found in dataset, using randomly generated gender data!")

    user_meta = pd.DataFrame(user_meta_data)
else:
    # If user_feat is not available, we must create synthetic data
    # This is just for demonstration - in practice, you'd need real data
    user_ids = list(range(dataset.user_num))
    np.random.seed(42)  # For reproducibility
    genders = np.random.choice(['M', 'F'], size=len(user_ids))

    user_meta = pd.DataFrame({
        uid_field: user_ids,
        'gender': genders
    })


    print("WARNING: No user features found in dataset, using synthetic gender data!")


# merge gender to error-df
err_df = err_df.merge(user_meta[[uid_field, 'gender']], on=uid_field)


# -------------- (1) group-error METRICS --------------------------
item_grp_err = _per_item_group_errors(err_df, 'gender', 'error')

g0_err = err_df.loc[err_df.gender == 1, 'error']
g1_err = err_df.loc[err_df.gender == 2, 'error']

group_error_metrics = {
    'MAE-gap': mae_gap(g0_err.abs(), g1_err.abs()),
    'U_val': value_unfairness(item_grp_err),
    'U_abs': absolute_unfairness(item_grp_err),
    'U_under': under_unfairness(item_grp_err),
    'U_over': over_unfairness(item_grp_err),
    'KS': ks_statistic(g0_err, g1_err)
}

# -------------- (2) user-hit utility (+variance/abs-diff) ---------
# ground-truth set of relevant items (rating >=0.75)
test_df = pd.DataFrame({
    uid_field: u_t.cpu().numpy(),
    iid_field: i_t.cpu().numpy(),
    'rating': r_t.cpu().numpy()
})

print(test_df)

rel_set = {(u, i) for u, i, r in test_df.itertuples(index=False) if r >= 0.75}


def hit_at_k(row):
    u = row[uid_field]
    recs = row.topk
    print(f"{u}, {recs}, {rel_set}")
    return int(any((u, i) in rel_set for i in recs))


topk_df = pandas.read_csv("output/CKE_top10.csv")
topk_df["topk"] = topk_df["topk"].apply(ast.literal_eval)

print(topk_df.iloc[0, :])

topk_df['hit'] = topk_df.apply(hit_at_k, axis=1)

print(topk_df.iloc[0, :])

males = user_meta[user_meta.gender == 1][uid_field]
females = user_meta[user_meta.gender == 2][uid_field]

hits_m = topk_df[topk_df[uid_field].isin(males)]['hit']
hits_f = topk_df[topk_df[uid_field].isin(females)]['hit']

utility_metrics = {
    'AbsDiff_hit': absolute_difference(hits_m, hits_f),
    'Var_hit': variance(topk_df['hit'])
}

# columns’ names that RecBole uses
uid_field = config['USER_ID_FIELD']
iid_field = config['ITEM_ID_FIELD']

# pull the (user,item) pairs from the training dataloader
train_int = pd.DataFrame({
    uid_field: dataset.inter_feat[uid_field].cpu().numpy(),
    iid_field: dataset.inter_feat[iid_field].cpu().numpy()
})

# -------------- (3) item-exposure fairness (pop vs tail) ----------
# popularity bins (median split)
pop_cnt = train_int[iid_field].value_counts()
median = pop_cnt.median()
pop_bin = {iid: int(cnt >= median) for iid, cnt in pop_cnt.items()}  # 0 tail, 1 pop

exp_counts = defaultdict(int)
for recs in topk_df.topk:
    for iid in recs:
        exp_counts['pop' if pop_bin.get(iid, 1) else 'tail'] += 1

item_metrics = {
    'Gini_exposure': gini(list(exp_counts.values())),
    'Exposure_tail': exp_counts['tail'],
    'Exposure_pop': exp_counts['pop']
}

# ================================================================
#  C)  Print consolidated block
# ================================================================
print("\n=== Group-error metrics (gender) ===")
for k, v in group_error_metrics.items():
    print(f"{k:15s} : {v:.4f}")

print(f"\n=== User-utility fairness (hit@{TOPK}) ===")
for k, v in utility_metrics.items():
    print(f"{k:15s} : {v:.4f}")

print("\n=== Item-exposure metrics (pop vs tail) ===")
for k, v in item_metrics.items():
    print(f"{k:15s} : {v}")

os.makedirs('fairness_outputs', exist_ok=True)
fairness_df = pd.DataFrame({
    'MAE-gap': [group_error_metrics['MAE-gap']],
    'U_val': [group_error_metrics['U_val']],
    'U_abs': [group_error_metrics['U_abs']],
    'U_under': [group_error_metrics['U_under']],
    'U_over': [group_error_metrics['U_over']],
    'KS': [group_error_metrics['KS']],
    'AbsDiff_hit': [utility_metrics['AbsDiff_hit']],
    'Var_hit': [utility_metrics['Var_hit']],
    'Gini_exposure': [item_metrics['Gini_exposure']],
    'Exposure_tail': [item_metrics['Exposure_tail']],
    'Exposure_pop': [item_metrics['Exposure_pop']],

})

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

fairness_df.to_csv(f'fairness_outputs/{current_time}')
#fairness_df.to_csv(f'fairness_outputs/1')
print("Fairness results saved -> ", f'fairness_outputs/{current_time}')