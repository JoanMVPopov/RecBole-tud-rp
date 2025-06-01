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

df_combined = pd.DataFrame()

for run in range(len(sweep_runs)):
    # if run == 0:
    #     continue
    # if run > 2:
    #     break

    #df = sweep_runs[run].history(keys=["epoch", "train/loss_1", "train/loss_2", "train/loss_3"], x_axis="epoch")
    df = sweep_runs[run].history(keys=eval_keys)
    #df = sweep_runs[run].history(keys=eval_keys).drop("_step", axis=1)
    #print(df)
    df_combined = pd.concat([df_combined, df])

print(df_combined)

# COMMON EVAL METRICS
for metric in metrics:
    if metric != "gauc":
        df_selected = df_combined[["eval/all/" + metric + f"@{topk}", "eval/male/" + metric + f"@{topk}", "eval/female/" + metric + f"@{topk}"]]
    else:
        df_selected = df_combined[["eval/all/" + metric, "eval/male/" + metric, "eval/female/" + metric]]

    plt.figure()
    df_selected.boxplot()
    print(df_selected)
    if metric != "gauc":
        plt.title(f"Boxplot for {metric}@{topk}")
    else:
        plt.title(f"Boxplot for {metric}")
    plt.show()

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