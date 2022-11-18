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

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

import transformers
import tokenizers
print(f'transformers.__version__: {transformers.__version__}')
print(f'tokenizers.__version__: {tokenizers.__version__}')
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
os.environ['TOKENIZERS_PARALLELISM']='true'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed=42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

class CFG:
    wandb=True
    competition='FB3'
    _wandb_kernel='nakama'
    debug=False
    apex=True
    print_freq=20
    num_workers=4
    model="microsoft/deberta-v3-base"
    gradient_checkpointing=True
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=4
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=8
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    pooling="mean"

train_file = './feedback-prize-english-language-learning/train.csv'
df_train = pd.read_csv(train_file)

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold    
Fold = MultilabelStratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
for n, (train_index, val_index) in enumerate(Fold.split(df_train, df_train[CFG.target_cols])):
    df_train.loc[val_index, 'fold'] = int(n)
df_train['fold'] = df_train['fold'].astype(int)
print(df_train.head())

# ensembles = np.load('exp040/fold_0_embedding.npy')
# print(ensembles.shape)

from sklearn.metrics import mean_squared_error


def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(6)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i], y_pred[:,i])))
    return np.mean(rmse_scores)

def embeddings():

    models = [
        'exp040',
        'exp041',
        # 'exp042',
        # 'exp043',
    ]
    train_embeddings = []
    valid_embeddings = []
    for m in models:
        fold = 0
        train_path = f'{m}/train_fold_{fold}_embedding.npy'
        train_embeddings.append(np.load(train_path))
        
        valid_path = f'{m}/valid_fold_{fold}_embedding.npy'
        valid_embeddings.append(np.load(valid_path))

    train_embeddings = np.concatenate(train_embeddings, axis=1)
    valid_embeddings = np.concatenate(valid_embeddings, axis=1)
    return train_embeddings, valid_embeddings

target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
preds = []
scores = []
def comp_score(y_true,y_pred):
    rmse_scores = []
    for i in range(len(target_cols)):
        rmse_scores.append(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i])))
    return np.mean(rmse_scores)

fold = 0 
train_folds = df_train[df_train['fold'] != fold].reset_index(drop=True)
valid_folds = df_train[df_train['fold'] == fold].reset_index(drop=True)
train_labels = train_folds[CFG.target_cols].values
valid_labels = valid_folds[CFG.target_cols].values



train_preds = np.zeros((len(target_cols), 6))
valid_preds = np.zeros((len(target_cols), 6))

train_embeddings, valid_embeddings = embeddings()
from sklearn.svm import SVR


for i, t in enumerate(target_cols):
    print(t,', ',end='')
    print(i)
    # clf = SVR(C=1)
    # clf.fit(train_embeddings, train_labels[:,i])
    # train_preds[:,i] = clf.predict(train_embeddings)
    # valid_preds[:,i] = clf.predict(valid_embeddings)

# score = comp_score(valid_labels, valid_preds)
# print(score)

# for fold in range(FOLDS):
#     print('#'*25)
#     print('### Fold',fold+1)
#     print('#'*25)
    
#     dftr_ = dftr[dftr["FOLD"]!=fold]
#     dfev_ = dftr[dftr["FOLD"]==fold]
    
#     tr_text_feats = all_train_text_feats[list(dftr_.index),:]
#     ev_text_feats = all_train_text_feats[list(dfev_.index),:]
    
#     ev_preds = np.zeros((len(ev_text_feats),6))
#     test_preds = np.zeros((len(te_text_feats),6))
#     for i,t in enumerate(target_cols):
#         print(t,', ',end='')
#         clf = SVR(C=1)
#         clf.fit(tr_text_feats, dftr_[t].values)
#         ev_preds[:,i] = clf.predict(ev_text_feats)
#         test_preds[:,i] = clf.predict(te_text_feats)
#     print()
#     score = comp_score(dfev_[target_cols].values,ev_preds)
#     scores.append(score)
#     print("Fold : {} RSME score: {}".format(fold,score))
#     preds.append(test_preds)
    
# print('#'*25)
# print('Overall CV RSME =',np.mean(scores))
