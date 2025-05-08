"""
Train a RecBole model and store each user’s Top-K list in a .csv
"""
import os, json, torch
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
config, model, dataset, trainer = run_recbole(
        model        = MODEL,
        dataset      = DATASET,
        config_file_list = [CONFIG]

)

# 2) full-sort scores  (RecBole helper) -------------------------------------
# user_ids, item_ids, scores = full_sort_topk(
#         model      = model,
#         test_data  = dataset,           # RecBole’s internal Datasets
#         k       = TOPK,
#         device     = trainer.device
# )

topk_scores, topk_index = full_sort_topk(
        uid_series=user_ids,
        model      = model,
        test_data  = dataset,           # RecBole’s internal Datasets
        k       = TOPK,
        device     = trainer.device
)

# 3) store as one row per user (uid, iid1 … iidK) ---------------------------
topk_df = pd.DataFrame({
    'user_id' : user_ids,
    'topk'    : [json.dumps(iids) for iids in topk_index]    # save list as str
})
os.makedirs('output', exist_ok=True)
topk_df.to_csv(f'output/{MODEL}_top{TOPK}.csv', index=False)
print("Top-K list saved -> ", f'output/{MODEL}_top{TOPK}.csv')
