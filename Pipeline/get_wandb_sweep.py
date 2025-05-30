import os

import wandb
import yaml

if __name__ == '__main__':
    with open('sweep.yaml') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="recbole-fairness-sweep",
        entity="yoankich-tu-delft-rp"
    )
