import pandas as pd
from matplotlib import pyplot as plt

import wandb

api = wandb.Api()
entity, project = "yoankich-tu-delft-rp", "recbole-fairness-sweep"

sweep = api.sweep(entity + "/" + project + "/6igigjav")
sweep_runs = sweep.runs

topk = 10
genders = ['all', 'female', 'male']
metrics = ["recall", "mrr", "ndcg", "hit", "map", "precision", "gauc",
    "tailpercentage", "giniindex", "popularitypercentage"]

metrics_subset_to_show = ["recall", "tailpercentage", "giniindex", "popularitypercentage"]

metrics_fairness_separated = ['giniindex', 'popularitypercentage']

metrics_all_only = ["Differential Fairness of sensitive attribute gender", "Value Unfairness of sensitive attribute gender",
    "Absolute Unfairness of sensitive attribute gender", "Underestimation Unfairness of sensitive attribute gender",
    "Overestimation Unfairness of sensitive attribute gender", "NonParity Unfairness of sensitive attribute gender"]

################
# EVAL ONLY
################

eval_keys = []

for gender in genders:
    for metric in metrics:
        if metric != "gauc":
            eval_keys.append(f"eval/{gender}/{metric}@{topk}")
        else:
            eval_keys.append(f"eval/{gender}/{metric}")

for metric in metrics_all_only:
    eval_keys.append("eval/all/" + metric)

print(eval_keys)
print(len(eval_keys))
print(len(sweep_runs))

df_combined = pd.DataFrame()

for run in range(len(sweep_runs)):
    df = sweep_runs[run].history(keys=eval_keys)
    #df = sweep_runs[run].history(keys=eval_keys).drop("_step", axis=1)
    df_combined = pd.concat([df_combined, df])

print(df_combined)


# SELECT TOP 5 BASED ON MAXIMIZED METRIC
df_combined = df_combined.sort_values(ascending=False, by=f'eval/all/recall@{topk}')[:5]

models = ["CKE", "CKFG", "RippleNet", "Random", "Pop", "ItemKNN"]

for metric in metrics_subset_to_show:
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
    for i, model in enumerate(models):
        if metric != "gauc":
            df_selected = df_combined[["eval/all/" + metric + f"@{topk}", "eval/male/" + metric + f"@{topk}", "eval/female/" + metric + f"@{topk}"]]
        else:
            df_selected = df_combined[["eval/all/" + metric, "eval/male/" + metric, "eval/female/" + metric]]

        ax = axes[i]
        ax.set_title(model)
        df_selected.columns = ["All", "Male", "Female"]
        df_selected.boxplot(ax=ax, widths=0.6)

    if metric != "gauc":
        fig.suptitle(f"Boxplot for {metric}@{topk}")
    else:
        fig.suptitle(f"Boxplot for {metric}")

    plt.tight_layout()
    plt.savefig(f"figures/{metric}_3split_boxplots_allmodels.png")


for metric in metrics_all_only:
    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))
    for i, model in enumerate(models):
        if metric != "gauc":
            df_selected = df_combined[["eval/all/" + metric + f"@{topk}", "eval/male/" + metric + f"@{topk}", "eval/female/" + metric + f"@{topk}"]]
        else:
            df_selected = df_combined[["eval/all/" + metric, "eval/male/" + metric, "eval/female/" + metric]]

        ax = axes[i]
        ax.set_title(model)
        df_selected.columns = ["All", "Male", "Female"]
        df_selected.boxplot(ax=ax, widths=0.6)

    if metric != "gauc":
        fig.suptitle(f"Boxplot for {metric}@{topk}")
    else:
        fig.suptitle(f"Boxplot for {metric}")

    plt.tight_layout()
    plt.savefig(f"figures/{metric}_3split_boxplots_allmodels.png")

# FAIRNESS METRICS FOR ALL
fig, axes = plt.subplots(nrows=1, ncols=len(metrics_all_only), figsize=(20, 5))

for i, metric in enumerate(metrics_all_only):
    ax = axes[i]
    column = "eval/all/" + metric
    df_combined[[column]].boxplot(ax=ax, widths=0.6)
    ax.set_title(metric)
    ax.set_ylabel("Value")
    ax.grid(True)

plt.tight_layout()
plt.show()