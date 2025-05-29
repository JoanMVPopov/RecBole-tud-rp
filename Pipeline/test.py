
import sys, os

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
#from recbole.model.general_recommender import BPR   # or import your MODEL
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.utils.case_study import full_sort_topk
from recbole.model.knowledge_aware_recommender.cke import CKE
from recbole.data.dataloader.knowledge_dataloader import KGDataLoaderState
from recbole.quick_start import run_recbole

if __name__ == "__main__":
    # config = Config(
    #
    #     model='CKE',
    #     dataset='ml-100k',
    # )

    print(torch.cuda.is_available())

    #run_recbole(model='CKE', dataset='ml-100k', config_file_list=['conf/model_kg.yaml'])