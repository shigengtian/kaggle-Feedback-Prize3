# ====================================================
# Directory settings
# ====================================================
# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=4
    path="exp025_cohesion/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-base"
    gradient_checkpointing=False
    batch_size=16
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]

import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

# os.system('pip uninstall -y transformers')
# os.system('pip uninstall -y tokenizers')
# os.system('python -m pip install --no-index --find-links=../input/fb3-pip-wheels transformers')
# os.system('python -m pip install --no-index --find-links=../input/fb3-pip-wheels tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================================
# tokenizer
# ====================================================
CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')

# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv('feedback-prize-english-language-learning/2021_pseduo_label.csv')
# submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')

print(f"test.shape: {test.shape}")
# print(f"submission.shape: {submission.shape}")

# sort by length to speed up inference
test['tokenize_length'] = [len(CFG.tokenizer(text)['input_ids']) for text in test['full_text'].values]
test = test.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        #max_length=CFG.max_len,
        #pad_to_max_length=True,
        #truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs

# # ====================================================
# # Model
# # ====================================================
# class MeanPooling(nn.Module):
#     def __init__(self):
#         super(MeanPooling, self).__init__()
        
#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings
    
# # ====================================================
# # Model
# # ====================================================
# class MeanPooling(nn.Module):
#     def __init__(self):
#         super(MeanPooling, self).__init__()
        
#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings
    

# class CustomModel(nn.Module):
#     def __init__(self, cfg, config_path=None, pretrained=False):
#         super().__init__()
#         self.cfg = cfg
#         self.config = torch.load(config_path)
#         if pretrained:
#             self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
#         else:
#             self.model = AutoModel.from_config(self.config)
#         if self.cfg.gradient_checkpointing:
#             self.model.gradient_checkpointing_enable()

#         self.dropout = nn.Dropout(0.1)
#         self.dropout1 = nn.Dropout(0.1)
#         self.dropout2 = nn.Dropout(0.2)
#         self.dropout3 = nn.Dropout(0.3)
#         self.dropout4 = nn.Dropout(0.4)
#         self.dropout5 = nn.Dropout(0.5)

#         self.pool = MeanPooling()
#         self.fc = nn.Linear(self.config.hidden_size, 6)
#         self._init_weights(self.fc)
        
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
        
#     def feature(self, inputs):
#         outputs = self.model(**inputs)
#         last_hidden_states = outputs[0]
#         feature = self.pool(last_hidden_states, inputs['attention_mask'])
#         return feature

#     def forward(self, inputs):
#         feature = self.feature(inputs)

#         logits1 = self.fc(self.dropout1(feature))
#         logits2 = self.fc(self.dropout2(feature))
#         logits3 = self.fc(self.dropout3(feature))
#         logits4 = self.fc(self.dropout4(feature))
#         logits5 = self.fc(self.dropout5(feature))

#         logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        
#         # output = self.fc(logits)
#         return logits

# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output

# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

def predict():
    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    predictions = []
    for col in ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']:
        pred_cols = []
        for fold in CFG.trn_fold:
            model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)

            state = torch.load(f"exp025_{col}/deberta-base-checkpoint-9750_fold{fold}_best.pth",
                               map_location=torch.device('cpu'))

            model.load_state_dict(state['model'])
            pred_col = inference_fn(test_loader, model, device)
            pred_cols.append(pred_col)
            del model, state, pred_col; gc.collect()
            torch.cuda.empty_cache()
            
        predictions.append(np.mean(pred_cols, axis=0))
        
    return predictions

predictions = predict()

cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

# _interval  = 0.01

for n in range(len(cols)):
#     p = predictions[n]
#     p[(p > 5-_interval)] = 5
#     p[(p < 1)] = 1
    
#     p[(p > 1) & (p < 1 + _interval)] = 1
#     p[(p > (1.5 - _interval)) & (p < (1.5 + _interval))] = 1.5
#     p[(p > (2 - _interval)) & (p < (2 + _interval))] = 2
#     p[(p > (2.5 - _interval)) & (p < (2.5 + _interval))] = 2.5
#     p[(p > (3 - _interval)) & (p < (3 + _interval))] = 3
#     p[(p > (3.5 - _interval)) & (p < (3.5 + _interval))] = 3.5
#     p[(p > (4 - _interval)) & (p < (4 + _interval))] = 4
#     p[(p > (4.5 - _interval)) & (p < (4.5 + _interval))] = 4.5
    
    test[cols[n]] = predictions[n]

test.to_csv('test.csv', index=False)