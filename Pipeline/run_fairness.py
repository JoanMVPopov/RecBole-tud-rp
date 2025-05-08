# -------------------------------------------------------------
#  Unified fairness evaluation for MovieLens-100K + RecBole model
# -------------------------------------------------------------
import json, torch, numpy as np, pandas as pd
from collections import defaultdict
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
MODEL    = 'cke'        # CKE model as specified
DATASET  = 'ml-100k'    # RecBole's MovieLens-100K alias
TOPK     = 10

# ============================================================
#  A)  Load model & data  (+ full predictions, top-k lists)
# ============================================================
config, model, dataset, _, _, _ = load_data_and_model(
        f'saved/CKE-May-07-2025_12-06-56.pth')

print(f"\n\n\nDATASET: {dataset}")

model.eval()
device = model.device

# # ---------- test interactions (point-wise) -------------------
# test_data = dataset.dataset['test']
# # Use correct field names based on the config
# uid_field = config['USER_ID_FIELD']
# iid_field = config['ITEM_ID_FIELD']
# rating_field = config['RATING_FIELD']
#
# u_t   = torch.tensor(test_data[uid_field].tolist(), device=device)
# i_t   = torch.tensor(test_data[iid_field].tolist(), device=device)
# r_t   = torch.tensor(test_data[rating_field].tolist(), device=device)

# ---------- test interactions (point-wise) -------------------
# Use correct field names based on the config
uid_field = config['USER_ID_FIELD']
iid_field = config['ITEM_ID_FIELD']
rating_field = config['RATING_FIELD']

# Get test interactions - RecBole typically uses a DataLoader to get test data
# Here we'll access the test interaction data directly from the dataset
test_data = dataset.inter_feat  # This gets all interaction features
# We can filter for test data if needed based on specific indicators in the dataset
# For now, we'll use all available interactions

u_t   = torch.tensor(test_data[uid_field].numpy(), device=device)
i_t   = torch.tensor(test_data[iid_field].numpy(), device=device)
r_t   = torch.tensor(test_data[rating_field].numpy(), device=device)

intr = Interaction({uid_field: u_t, iid_field: i_t})
with torch.no_grad():
    preds = model.predict(intr).cpu().numpy()

err_signed = (r_t - preds).cpu().numpy()          # signed error

err_df = pd.DataFrame({
    uid_field: u_t.cpu(), iid_field: i_t.cpu(), 'error': err_signed
})

# ---------- full-sort top-k per user --------------------------
# Get user and item counts from the dataset
user_num = dataset.user_num
item_num = dataset.item_num
batch_size = 2048

topk_dict = {}
for start in tqdm(range(0, user_num, batch_size), desc='Full-sort'):
    end = min(start+batch_size, user_num)
    uids = torch.arange(start, end, device=device)
    repeated_u = uids.repeat_interleave(item_num)
    repeated_i = torch.arange(item_num, device=device).repeat(end-start)

    inter = Interaction({uid_field: repeated_u, iid_field: repeated_i})
    with torch.no_grad():
        scores = model.predict(inter).view(end-start, item_num)

    _, topk_idx = torch.topk(scores, k=TOPK, dim=-1)
    for offset, u in enumerate(uids.cpu().numpy()):
        topk_dict[int(u)] = topk_idx[offset].cpu().numpy().tolist()

topk_df = pd.DataFrame({uid_field: list(topk_dict.keys()),
                        'topk': list(topk_dict.values())})

# -------------------------------------------------------------
#  Unified fairness evaluation for MovieLens-100K + RecBole model
# -------------------------------------------------------------
import json, torch, numpy as np, pandas as pd
from collections import defaultdict
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
MODEL = 'cke'  # CKE model as specified
DATASET = 'ml-100k'  # RecBole's MovieLens-100K alias
TOPK = 10

# ============================================================
#  A)  Load model & data  (+ full predictions, top-k lists)
# ============================================================
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    f'saved/CKE-May-07-2025_12-06-56.pth')

print(f"\n\n\nDATASET: {DATASET}")

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

# ---------- full-sort top-k per user --------------------------
# Get user and item counts from the dataset
user_num = dataset.user_num
item_num = dataset.item_num
batch_size = 2048

topk_dict = {}
for start in tqdm(range(0, user_num, batch_size), desc='Full-sort'):
    end = min(start + batch_size, user_num)
    uids = torch.arange(start, end, device=device)
    repeated_u = uids.repeat_interleave(item_num)
    repeated_i = torch.arange(item_num, device=device).repeat(end - start)

    inter = Interaction({uid_field: repeated_u, iid_field: repeated_i})
    with torch.no_grad():
        scores = model.predict(inter).view(end - start, item_num)

    _, topk_idx = torch.topk(scores, k=TOPK, dim=-1)
    for offset, u in enumerate(uids.cpu().numpy()):
        topk_dict[int(u)] = topk_idx[offset].cpu().numpy().tolist()

topk_df = pd.DataFrame({uid_field: list(topk_dict.keys()),
                        'topk': list(topk_dict.values())})

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

# 2) Get training interactions from the dataset
# Use the train_data loader or the full dataset's inter_feat
if hasattr(train_data, 'dataset') and hasattr(train_data.dataset, 'inter_feat'):
    # Get interactions from training data
    train_uid = train_data.dataset.inter_feat[uid_field].numpy()
    train_iid = train_data.dataset.inter_feat[iid_field].numpy()

    # Create DataFrame from training interactions
    train_int = pd.DataFrame({
        uid_field: train_uid,
        iid_field: train_iid
    })
else:
    # Use the full dataset's interactions as a fallback
    train_uid = dataset.inter_feat[uid_field].numpy()
    train_iid = dataset.inter_feat[iid_field].numpy()

    train_int = pd.DataFrame({
        uid_field: train_uid,
        iid_field: train_iid
    })
    print("WARNING: Using all interactions instead of just training interactions!")

# gender 0/1 map
gender_map = {'M': 0, 'F': 1}
user_meta['g'] = user_meta.gender.map(gender_map)

# merge gender to error-df
err_df = err_df.merge(user_meta[[uid_field, 'g']], on=uid_field)

# -------------- (1) group-error METRICS --------------------------
item_grp_err = _per_item_group_errors(err_df, 'g', 'error')

g0_err = err_df.loc[err_df.g == 0, 'error']
g1_err = err_df.loc[err_df.g == 1, 'error']

group_error_metrics = {
    'MAE-gap': mae_gap(g0_err.abs(), g1_err.abs()),
    'U_val': value_unfairness(item_grp_err),
    'U_abs': absolute_unfairness(item_grp_err),
    'U_under': under_unfairness(item_grp_err),
    'U_over': over_unfairness(item_grp_err),
    'KS': ks_statistic(g0_err, g1_err)
}

# -------------- (2) user-hit utility (+variance/abs-diff) ---------
# ground-truth set of relevant items (rating >=4)
test_df = pd.DataFrame({
    uid_field: u_t.cpu().numpy(),
    iid_field: i_t.cpu().numpy(),
    'rating': r_t.cpu().numpy()
})
rel_set = {(u, i) for u, i, r in test_df.itertuples(index=False) if r >= 4}


def hit_at_k(row):
    u = row[uid_field]
    recs = row.topk
    return int(any((u, i) in rel_set for i in recs))


topk_df['hit'] = topk_df.apply(hit_at_k, axis=1)

males = user_meta[user_meta.g == 0][uid_field]
females = user_meta[user_meta.g == 1][uid_field]

hits_m = topk_df[topk_df[uid_field].isin(males)]['hit']
hits_f = topk_df[topk_df[uid_field].isin(females)]['hit']

utility_metrics = {
    'AbsDiff_hit': absolute_difference(hits_m, hits_f),
    'Var_hit': variance(topk_df['hit'])
}

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

# # =================================================================
# #  B)  Attach meta (gender, popularity)  +  compute fairness stats
# # =================================================================
# # ---------- load MovieLens-100K user / train interaction meta -------
#
# # After your load_data_and_model() call:
# data_path = config['data_path']      # usually "./dataset"
# dataset_name = config['dataset']     # "ml-100k"
# base = os.path.join(data_path, dataset_name)
#
# # 1) user metadata:
# # For ml-100k, load the user data
# user_meta = pd.read_csv(
#     os.path.join(base, f"{dataset_name}.user"),
#     sep="\t",
#     names=[   # the MovieLens-100K schema
#         uid_field,
#         'gender', 'age', 'occupation', 'zip_code'
#     ],
# )
#
# # 2) training interactions (to compute item‐popularity bins):
# train_int = pd.read_csv(
#     os.path.join(base, f"{dataset_name}.inter"),
#     sep="\t",
#     names=[
#         uid_field,
#         iid_field,
#         rating_field,
#         config['TIME_FIELD'],
#     ],
# )
#
# # gender 0/1 map
# gender_map = {'M': 0, 'F': 1}
# user_meta['g'] = user_meta.gender.map(gender_map)
#
# # merge gender to error-df
# err_df = err_df.merge(user_meta[[uid_field, 'g']], on=uid_field)
#
# # -------------- (1) group-error METRICS --------------------------
# item_grp_err = _per_item_group_errors(err_df, 'g', 'error')
#
# g0_err = err_df.loc[err_df.g==0, 'error']
# g1_err = err_df.loc[err_df.g==1, 'error']
#
# group_error_metrics = {
#     'MAE-gap': mae_gap(g0_err.abs(), g1_err.abs()),
#     'U_val': value_unfairness(item_grp_err),
#     'U_abs': absolute_unfairness(item_grp_err),
#     'U_under': under_unfairness(item_grp_err),
#     'U_over': over_unfairness(item_grp_err),
#     'KS': ks_statistic(g0_err, g1_err)
# }
#
# # -------------- (2) user-hit utility (+variance/abs-diff) ---------
# # ground-truth set of relevant items (rating >=4)
# test_df = pd.DataFrame({
#     uid_field: u_t.cpu(),
#     iid_field: i_t.cpu(),
#     'rating': r_t.cpu()
# })
# rel_set = {(u, i) for u, i, r in test_df.itertuples(index=False) if r >= 4}
#
# def hit_at_k(row):
#     u = row[uid_field]
#     recs = row.topk
#     return int(any((u, i) in rel_set for i in recs))
#
# topk_df['hit'] = topk_df.apply(hit_at_k, axis=1)
#
# males = user_meta[user_meta.g == 0][uid_field]
# females = user_meta[user_meta.g == 1][uid_field]
#
# hits_m = topk_df[topk_df[uid_field].isin(males)]['hit']
# hits_f = topk_df[topk_df[uid_field].isin(females)]['hit']
#
# utility_metrics = {
#     'AbsDiff_hit': absolute_difference(hits_m, hits_f),
#     'Var_hit': variance(topk_df['hit'])
# }
#
# # -------------- (3) item-exposure fairness (pop vs tail) ----------
# # popularity bins (median split)
# pop_cnt = train_int[iid_field].value_counts()
# median = pop_cnt.median()
# pop_bin = {iid: int(cnt >= median) for iid, cnt in pop_cnt.items()}  # 0 tail, 1 pop
#
# exp_counts = defaultdict(int)
# for recs in topk_df.topk:
#     for iid in recs:
#         exp_counts['pop' if pop_bin.get(iid, 1) else 'tail'] += 1
#
# item_metrics = {
#     'Gini_exposure': gini(list(exp_counts.values())),
#     'Exposure_tail': exp_counts['tail'],
#     'Exposure_pop': exp_counts['pop']
# }
#
# # ================================================================
# #  C)  Print consolidated block
# # ================================================================
# print("\n=== Group-error metrics (gender) ===")
# for k, v in group_error_metrics.items():
#     print(f"{k:15s} : {v:.4f}")
#
# print(f"\n=== User-utility fairness (hit@{TOPK}) ===")
# for k, v in utility_metrics.items():
#     print(f"{k:15s} : {v:.4f}")
#
# print("\n=== Item-exposure metrics (pop vs tail) ===")
# for k, v in item_metrics.items():
#     print(f"{k:15s} : {v}")