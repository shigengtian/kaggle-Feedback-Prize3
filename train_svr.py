import lightgbm as lgb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.svm import SVR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig
import tokenizers
import transformers
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD, AdamW
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import scipy as sp
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import datetime
import math
import string
import pickle
import random
import joblib
import itertools
from distutils.util import strtobool
import warnings
warnings.filterwarnings('ignore')


print(f'transformers.__version__: {transformers.__version__}')
print(f'tokenizers.__version__: {tokenizers.__version__}')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)


class CFG:
    wandb = True
    competition = 'FB3'
    _wandb_kernel = 'nakama'
    debug = False
    apex = True
    print_freq = 20
    num_workers = 4
    model = "microsoft/deberta-v3-base"
    gradient_checkpointing = True
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 4
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 8
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_cols = ['cohesion', 'syntax', 'vocabulary',
                   'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True
    pooling = "mean"


# train_file = './feedback-prize-english-language-learning/train.csv'
train_file = './feedback-prize-english-language-learning/train_text_feature.csv'
df_train = pd.read_csv(train_file)

Fold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(df_train, df_train[CFG.target_cols])):
    df_train.loc[val_index, 'fold'] = int(n)
df_train['fold'] = df_train['fold'].astype(int)
print(df_train.head())
print(df_train.columns)

feature_column = ['full_text_num_words',
                  'full_text_num_unique_words', 'full_text_num_chars',
                  'full_text_num_stopwords', 'full_text_num_punctuations',
                  'full_text_num_words_upper', 'full_text_num_words_title',
                  'full_text_mean_word_len', 'full_text_num_paragraphs',
                  'full_text_num_contractions', 'full_text_polarity',
                  'full_text_subjectivity', 'full_text_nn_count', 'full_text_pr_count',
                  'full_text_vb_count', 'full_text_jj_count', 'full_text_uh_count',
                  'full_text_cd_count']


def comp_score(y_true, y_pred):
    rmse_scores = []
    for i in range(len(6)):
        rmse_scores.append(
            np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    return np.mean(rmse_scores)


def get_embeddings(fold):

    models = [
        'exp040',
        'exp041',
        'exp042',
        'exp043',
    ]
    train_embeddings = []
    valid_embeddings = []
    for m in models:
        train_path = f'{m}/train_fold_{fold}_embedding.npy'
        train_embeddings.append(np.load(train_path))

        valid_path = f'{m}/valid_fold_{fold}_embedding.npy'
        valid_embeddings.append(np.load(valid_path))

    train_embeddings = np.concatenate(train_embeddings, axis=1)
    valid_embeddings = np.concatenate(valid_embeddings, axis=1)
    return train_embeddings, valid_embeddings


target_cols = ['cohesion', 'syntax', 'vocabulary',
               'phraseology', 'grammar', 'conventions']


def comp_score(y_true, y_pred):
    rmse_scores = []
    for i in range(len(target_cols)):
        rmse_scores.append(
            np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    return np.mean(rmse_scores)


params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.01,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": -1,
          'metric': 'rmse',
          "random_state": 42,
          # 'device': 'gpu'
          }

# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.01,
#     'seed': 42,
#     'max_depth': -1,
#     'min_data_in_leaf': 10,
#     'verbosity': -1,
# }


oof = np.zeros((len(df_train), 6))

for fold in range(5):
    num_round = 5000
    print(f"fold ------{fold} ------")
    # df_train = df_train.reset_index(drop=True)
    train_index = df_train[df_train['fold'] != fold].index
    valid_index = df_train[df_train['fold'] == fold].index

    train_folds = df_train.loc[train_index]
    valid_folds = df_train.loc[valid_index]
    train_labels = train_folds[CFG.target_cols].values
    valid_labels = valid_folds[CFG.target_cols].values

    train_preds = np.zeros((len(train_labels), 6))
    valid_preds = np.zeros((len(valid_labels), 6))

    train_feature = train_folds[feature_column].values
    valid_feature = valid_folds[feature_column].values

    train_embeddings, valid_embeddings = get_embeddings(fold)

    train_embeddings = np.concatenate(
        [train_feature, train_embeddings], axis=1)
    valid_embeddings = np.concatenate(
        [valid_feature, valid_embeddings], axis=1)
    for i, t in enumerate(target_cols):
        print(t)
        trn_data = lgb.Dataset(train_embeddings, label=train_labels[:, i])
        val_data = lgb.Dataset(valid_embeddings, label=valid_labels[:, i])
        clf = lgb.train(params,
                        trn_data,
                        num_round,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=200,
                        early_stopping_rounds=200)
        p = clf.predict(valid_embeddings, num_iteration=clf.best_iteration)
        valid_preds[:, i] = p
    oof[valid_index] = valid_preds
    score = comp_score(valid_labels, valid_preds)
    print(score)

print(oof)
score = comp_score(df_train[target_cols].values, oof)
print(score)
