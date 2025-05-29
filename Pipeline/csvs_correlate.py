import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def csvs_correlate():
    fairness_metrics = ['absdiff_hit', 'exposure_pop', 'exposure_tail', 'gini', 'ks',
                        'mae_gap', 'u_abs', 'u_over', 'u_under', 'u_val', 'var_err']

    top_performer_number = "-17"
    bottom_performer_number = "-10"

    merged_fairness_df_top = pd.DataFrame()
    merged_fairness_df_bottom = pd.DataFrame()

    current_path = os.path.dirname(os.path.abspath(__file__))

    # Fairness csvs loop
    for filename in os.listdir(os.path.join(current_path, 'csvs')):
        if filename.endswith('.csv') and filename.split('_wandb')[0] in fairness_metrics:
            file_path = os.path.join('csvs', filename)
            df = pd.read_csv(file_path)

            for column in df.columns:
                if top_performer_number in column and ("MAX" not in column and "MIN" not in column):
                    df_current_name = column
                    new_name = top_performer_number[1:] + "_" + filename.split('_wandb')[0]
                    df.rename(columns={df_current_name: new_name}, inplace=True)
                    merged_fairness_df_top = pd.concat([merged_fairness_df_top, df[new_name]], axis=1)
                elif bottom_performer_number in column and ("MAX" not in column and "MIN" not in column):
                    df_current_name = column
                    new_name = bottom_performer_number[1:] + "_" + filename.split('_wandb')[0]
                    df.rename(columns={df_current_name: new_name}, inplace=True)
                    merged_fairness_df_bottom = pd.concat([merged_fairness_df_bottom, df[new_name]], axis=1)

    ###################################################################

    print(merged_fairness_df_top)
    print(merged_fairness_df_bottom)

    valid_acc_metrics = ['hit', 'mrr', 'ndcg', 'precision', 'recall']

    merged_valid_accuracy_df_top = pd.DataFrame()
    merged_valid_accuracy_df_bottom = pd.DataFrame()

    # Valid accuracy csvs loop
    for filename in os.listdir(os.path.join(current_path, 'csvs')):
        if filename.endswith('.csv') and filename.split('_wandb')[0] in valid_acc_metrics:
            file_path = os.path.join('csvs', filename)
            df = pd.read_csv(file_path)

            for column in df.columns:
                if top_performer_number in column and ("MAX" not in column and "MIN" not in column):
                    df_current_name = column
                    new_name = top_performer_number[1:] + "_" + filename.split('_wandb')[0]
                    df.rename(columns={df_current_name: new_name}, inplace=True)
                    merged_valid_accuracy_df_top = pd.concat([merged_valid_accuracy_df_top, df[new_name]], axis=1)
                elif bottom_performer_number in column and ("MAX" not in column and "MIN" not in column):
                    df_current_name = column
                    print(df_current_name)
                    new_name = bottom_performer_number[1:] + "_" + filename.split('_wandb')[0]
                    df.rename(columns={df_current_name: new_name}, inplace=True)
                    merged_valid_accuracy_df_bottom = pd.concat([merged_valid_accuracy_df_bottom, df[new_name]], axis=1)

    print(merged_valid_accuracy_df_top)
    print(merged_valid_accuracy_df_bottom)

    ##################################################################

    # after you’ve built your merged DataFrames...
    plot_fairness_accuracy_correlation(
        merged_fairness_df_top,
        merged_valid_accuracy_df_top,
        title="Top‐Performer Correlations"
    )

    plot_fairness_accuracy_correlation(
        merged_fairness_df_bottom,
        merged_valid_accuracy_df_bottom,
        title="Bottom‐Performer Correlations"
    )

    # # Compute correlation matrix between fairness (rows) and accuracy (columns)
    # correlation_matrix = merged_fairness_df_top.corrwith(
    #     merged_valid_accuracy_df_top, axis=0, method='pearson'
    # )
    #
    # # Since corrwith returns a Series, for a full matrix:
    # full_corr_matrix = pd.DataFrame(index=merged_fairness_df_top.columns, columns=merged_valid_accuracy_df_top.columns)
    #
    # for fairness_col in merged_fairness_df_top.columns:
    #     for acc_col in merged_valid_accuracy_df_top.columns:
    #         corr = merged_fairness_df_top[fairness_col].corr(merged_valid_accuracy_df_top[acc_col])
    #         full_corr_matrix.loc[fairness_col, acc_col] = corr
    #
    # full_corr_matrix = full_corr_matrix.astype(float)
    #
    # # Plot the heatmap
    # plt.figure(figsize=(10, 6))
    # sns.heatmap(full_corr_matrix, annot=True, fmt=".2f", cmap='bwr', center=0)
    # plt.title("Correlation: Fairness Metrics vs Accuracy Metrics (Top Performers)")
    # plt.ylabel("Fairness Metrics")
    # plt.xlabel("Accuracy Metrics")
    # plt.tight_layout()
    # plt.show()

def plot_fairness_accuracy_correlation(fair_df, acc_df, title="Something"):
    # align on index so we only compare matching rows
    fair_df, acc_df = fair_df.align(acc_df, join='inner', axis=0)

    data = np.concatenate([fair_df.values, acc_df.values], axis=1)
    corr_all = np.corrcoef(data, rowvar=False)
    m, k = fair_df.shape[1], acc_df.shape[1]
    corr_block = corr_all[:m, m:]

    fig, ax = plt.subplots(figsize=(k*1.2, m*0.5))
    im = ax.imshow(corr_block, cmap='bwr', vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(m))
    ax.set_xticklabels(acc_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(fair_df.columns)

    for i in range(m):
        for j in range(k):
            ax.text(j, i, f"{corr_block[i,j]:.2f}",
                    ha='center', va='center', fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    csvs_correlate()