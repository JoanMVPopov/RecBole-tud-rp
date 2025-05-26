"""
Train a RecBole model and store each user’s Top-K list in a .csv
"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from recbole.quick_start import run_recbole
from recbole.utils.case_study import full_sort_topk

DATASET  = 'ml-100k'      # folder name
MODEL    = 'CKE'        # change to CKE / KGAT / … (see cheat-sheet)
CONFIG   = 'conf/model_kg.yaml'
TOPK     = 10

# 1) train & get objects back ----------------------------------------------
result = run_recbole(
        model        = MODEL,
        dataset      = DATASET,
        config_file_list = [CONFIG]
)

config = result["config"]
model = result["model"]
trainer = result["trainer"]
best_valid_score = result["best_valid_score"]
valid_score_bigger = result["valid_score_bigger"]
best_valid_result = result["best_valid_result"]
test_result = result["test_result"]

test_dataset = None
import pickle
with open("split_data.pth", 'rb') as f:
    test_dataset = pickle.load(f)

# 2) full-sort scores  (RecBole helper) -------------------------------------
# user_ids, item_ids, scores = full_sort_topk(
#         model      = model,
#         test_data  = dataset,           # RecBole’s internal Datasets
#         k       = TOPK,
#         device     = trainer.device
# )

# uid_series=train_dataset.dataset["user_id"].unique()

uid_col    = test_dataset.dataset.uid_field            # e.g. 'user_id'
uid_tensor = test_dataset.dataset.inter_feat[uid_col]  # 1-D torch tensor

# (optional) keep each user only once
uid_tensor = torch.unique(uid_tensor)

# → NumPy ndarray
uid_array = uid_tensor.cpu().numpy()      # dtype: int64

topk_scores, topk_index = full_sort_topk(
        uid_series=uid_array,
        model      = model,
        test_data  = test_dataset,           # RecBole’s internal Datasets
        k       = TOPK,
        device     = trainer.device
)

# 3) store as one row per user (uid, iid1 … iidK) ---------------------------
topk_df = pd.DataFrame({
    'user_id' : uid_array.tolist(),
    'topk'    : topk_index.cpu().numpy().tolist()    # save list as str
})

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs('output', exist_ok=True)
topk_df.to_csv(f'output/{MODEL}_top{TOPK}_{current_time}.csv', index=False)
print("Top-K list saved -> ", f'output/{MODEL}_top{TOPK}_{current_time}.csv')
