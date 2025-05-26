import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm

from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# RecBole imports
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR   # or import your MODEL
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils.case_study import full_sort_topk

# your fairness‐metrics module
from fairness.metrics import (
    _per_item_group_errors, mae_gap, value_unfairness,
    absolute_unfairness, under_unfairness, over_unfairness, ks_statistic,
    absolute_difference, variance, gini, ks_area_difference
)


def evaluate_fairness(config, model, dataset, train_data, test_data, K=10):
    """Compute all three blocks of fairness metrics for one model snapshot."""
    device = model.device
    model.eval()

    # --- full‐sort top‐K on test set
    uid_col = test_data.dataset.uid_field
    uid_array = torch.unique(
        test_data.dataset.inter_feat[uid_col]
    ).cpu().numpy()
    _, topk_index = full_sort_topk(
        uid_series=uid_array,
        model=model,
        test_data=test_data,
        k=K,
        device=device
    )
    topk_lists = topk_index.cpu().numpy().tolist()
    topk_df = pd.DataFrame({ 'user_id': uid_array, 'topk': topk_lists })

    # --- point‐wise errors
    # read the field names from your config
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    rating_field = config['RATING_FIELD']

    # inter_dict = dataset.inter_feat.cpu().numpy()
    #
    # df_interactions = pd.DataFrame(inter_dict)
    #
    # # 3) now you can print, slice, describe, etc.
    # print(df_interactions.head())
    # print(df_interactions.shape)
    # print(df_interactions['rating'].describe())
    #
    # print(test_data.dataset)

    u_t = test_data.dataset.inter_feat[uid_field].to(device)
    i_t = test_data.dataset.inter_feat[iid_field].to(device)
    r_t = test_data.dataset.inter_feat[rating_field].to(device)

    from recbole.data.interaction import Interaction
    intr = Interaction({uid_field: u_t, iid_field: i_t})
    with torch.no_grad():
        preds = model.predict(intr).cpu().numpy()
    errors = (r_t.cpu().numpy() - preds)
    #errors_pred_true = (preds - u_t.cpu().numpy())

    err_df = pd.DataFrame({
        uid_field: u_t.cpu().numpy(),
        iid_field: i_t.cpu().numpy(),
        'error': errors
        #'error_pred_true': errors_pred_true
    })

    # FOR USER-LEVEL VARIANCE, this is the f(v) where v is the user
    err_df['abs_error'] = err_df['error'].abs()
    user_error_means = err_df.groupby(uid_field)['abs_error'].mean().values

    # --- user meta (assumes gender in user_feat; else inject synthetic for demo)
    if hasattr(dataset, 'user_feat') and dataset.user_feat is not None and 'gender' in dataset.user_feat:
        user_meta = pd.DataFrame({
            uid_field: dataset.user_feat[uid_field].numpy(),
            'gender': dataset.user_feat['gender'].numpy()
        })
    else:
        # fallback: 50/50 random genders
        n_users = dataset.user_num
        np.random.seed(42)
        user_meta = pd.DataFrame({
            uid_field: np.arange(n_users),
            'gender': np.random.choice([1, 2], size=n_users)
        })

    err_df = err_df.merge(user_meta, on=uid_field)
    item_grp_err = _per_item_group_errors(err_df, 'gender', 'error')

    # after you’ve merged user_meta into err_df
    groups = sorted(err_df['gender'].unique())
    if len(groups) == 2:
        g0, g1 = groups
        errors0 = err_df.loc[err_df.gender == g0, 'error']
        errors1 = err_df.loc[err_df.gender == g1, 'error']
        #ks = ks_statistic(errors0, errors1)
        ks = ks_area_difference(
            errors0.values,  # or .values
            errors1.values,
            n_bins=100  # or whatever resolution you like
        )
    else:
        # fallback if only one group present
        ks = np.nan

    group_error = {
        'MAE_gap': abs(errors0.abs().mean() - errors1.abs().mean()),
        'U_val': value_unfairness(item_grp_err),
        'U_abs': absolute_unfairness(item_grp_err),
        'U_under': under_unfairness(item_grp_err),
        'U_over': over_unfairness(item_grp_err),
        'KS': ks
    }

    # --- user‐utility hit@K
    # build relevance set
    rel = set((u,i) for u,i,r in zip(
        u_t.cpu().numpy(), i_t.cpu().numpy(), r_t.cpu().numpy()
    ) if r >= 0.75)
    def hit(u, recs):
        return int(any((u,i) in rel for i in recs))
    topk_df['hit'] = topk_df.apply(lambda row: hit(row['user_id'], row['topk']), axis=1)

    males = user_meta[user_meta.gender == 1][uid_field]
    females = user_meta[user_meta.gender == 2][uid_field]

    hits_m = topk_df[topk_df.user_id.isin(males)]['hit']
    hits_f = topk_df[topk_df.user_id.isin(females)]['hit']

    utility = {
        'AbsDiff_hit': absolute_difference(hits_m, hits_f),
        'Var_err': variance(user_error_means)
    }

    ##################
    ## ITEM EXPOSURE
    ##################

    # --- item‐exposure pop vs tail (use the RecBole Dataset’s inter_feat for global popularity)
    # We use dataset.inter_feat to get interaction counts from the whole dataset,
    all_interactions_df = pd.DataFrame({
        iid_field: dataset.inter_feat[iid_field].cpu().numpy()
    })

    # Count interactions per item. value_counts() sorts by frequency in descending order by default.
    item_interaction_counts = all_interactions_df[
        iid_field].value_counts()

    pop_bin = {}

    if not item_interaction_counts.empty:
        # 2. Calculate total interactions
        total_interactions = item_interaction_counts.sum()

        # 3. Calculate cumulative sum of interactions
        #    The index of item_interaction_counts are the item_ids, sorted by popularity
        cumulative_interactions = item_interaction_counts.cumsum()

        # 4. Define the 80% threshold for interactions
        interaction_threshold = 0.80 * total_interactions

        # This might also work, the method below works fine too
        # 5. Identify "pop" items: those items whose cumulative sum of interactions
        #    falls within the first 80% of total interactions.
        # pop_item_ids_series = cumulative_interactions[cumulative_interactions <= interaction_threshold]
        # pop_item_ids = set(pop_item_ids_series.index)

        # Calculate how many items needed to reach 80% of interactions
        running_sum = 0
        pop_item_ids = set()
        for item_id, count in item_interaction_counts.items():
            running_sum += count
            pop_item_ids.add(item_id)
            if running_sum >= interaction_threshold:
                break

        # For logging/verification (optional)
        num_pop_items = len(pop_item_ids)
        if dataset.item_num > 0:
            percentage_pop_items = (num_pop_items / dataset.item_num) * 100
            actual_interactions_by_pop = item_interaction_counts[item_interaction_counts.index.isin(pop_item_ids)].sum()
            percentage_interactions_by_pop = (
                                                         actual_interactions_by_pop / total_interactions) * 100 if total_interactions > 0 else 0
            print(f"Pop/Tail Split: Identified {num_pop_items} pop items ({percentage_pop_items:.2f}% of total items), "
                  f"accounting for {percentage_interactions_by_pop:.2f}% of total interactions.")
        else:
            print("Pop/Tail Split: No items in dataset to calculate percentages.")

        # 6. Create pop_bin: 1 for pop, 0 for tail
        #    Iterate through all possible item IDs in the dataset (0 to n_items-1)
        #    RecBole item IDs are typically 0-indexed integers.
        for iid_val in range(dataset.item_num):  # dataset.item_num gives the total number of unique items
            if iid_val in pop_item_ids:
                pop_bin[iid_val] = 1  # Pop
            else:
                pop_bin[iid_val] = 0  # Tail
    else:
        # Handle the case where there are no interactions (e.g., empty dataset split)
        # In this scenario, all items are effectively "tail" or undefined.
        # We'll define all as tail.
        print("No interactions found in the provided data. All items considered tail for exposure calculation.")
        for iid_val in range(dataset.item_num):
            pop_bin[iid_val] = 0  # All tail

    # 7. Calculate exposure based on topk_lists
    exp = defaultdict(int)
    for recs_for_user in topk_lists:  # topk_lists is a list of lists [[item1, item2,..], [item_x, item_y,..]]
        for iid_in_recs in recs_for_user:
            # Use .get(iid_in_recs, 0) to default to tail (0) if an item_id is somehow
            # not in pop_bin (e.g., if item_ids in topk_lists go beyond dataset.item_num, which shouldn't happen).
            if pop_bin.get(iid_in_recs, 0) == 1:
                exp['pop'] += 1
            else:
                exp['tail'] += 1

    # Ensure Gini is calculated only if there are at least two categories or non-zero exposures
    gini_value = 0.0
    exposure_values = [count for count in exp.values() if count > 0]  # Get non-zero exposure counts
    if len(exposure_values) > 1:
        gini_value = gini(exposure_values)
    elif len(exposure_values) == 1 and exposure_values[0] > 0:  # if only one group has all exposure
        gini_value = gini([exp.get('pop', 0), exp.get('tail', 0)])

    item_exp = {
        'Gini_exposure': gini_value,
        'Exposure_tail': exp.get('tail', 0),
        'Exposure_pop': exp.get('pop', 0)
    }

    # merge all
    metrics = { **group_error, **utility, **item_exp }
    return metrics


if __name__ == '__main__':
    # ---------------------------
    # 1) config & data prep
    # ---------------------------
    config = Config(
        config_file_list=['conf/model_kg.yaml'],
        model = 'CKE',
        dataset = 'ml-100k'
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ---------------------------
    # 2) model & trainer
    # ---------------------------
    # model = BPR(config, train_data.dataset).to(config['device'])  # or replace with your MODEL
    # trainer = Trainer(config, model)

    from recbole.model.knowledge_aware_recommender.cke import CKE
    model = CKE(config, dataset).to(config['device'])
    trainer = Trainer(config, model)

    max_epoch = config['epochs']
    history = defaultdict(list)

    from recbole.data.dataloader.knowledge_dataloader import KGDataLoaderState

    for epoch in range(1, max_epoch + 1):
        print(f"\n>>> Epoch {epoch}/{max_epoch}")

        # — TRAIN —
        train_data.set_mode(KGDataLoaderState.RSKG)
        train_loss = trainer._train_epoch(train_data, epoch)

        # — VALIDATE (only if your loader supports it) —
        if hasattr(valid_data, 'set_mode'):
            valid_data.set_mode(KGDataLoaderState.RS)
            valid_result = trainer._valid_epoch(valid_data, epoch)

        # — FAIRNESS —
        metrics = evaluate_fairness(config, model, dataset, train_data, test_data, K=10)
        for name, val in metrics.items():
            history[name].append(val)

    # ---------------------------
    # 4) plot
    # ---------------------------
    epochs = list(range(1, max_epoch + 1))
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for metric, values in history.items():
        plt.figure(figsize=(10, 6))

        # 1) draw the line (with label)
        plt.plot(epochs, values, label=metric)

        # 2) now set labels & title
        plt.xlabel('Epoch')
        plt.ylabel('Metric value')
        plt.title(f'{metric} over training epochs')

        # 3) and finally build the legend *after* the plot exists
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f'figures/{metric}_{current_time}.png')
        plt.close()

    # plt.show()
