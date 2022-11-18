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


import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
# %env TOKENIZERS_PARALLELISM=true
import argparse
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, default="exp038/")
    parser.add_argument("--exp_no", type=str, required=False, default="exp038")
    parser.add_argument("--output", type=str, default=".", required=False)
    # parser.add_argument("--batch_size", type=int, default=2, required=False)
    # parser.add_argument("--valid_batch_size", type=int, default=16, required=False)
    return parser.parse_args()


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
        self.model = AutoModel.from_config(self.config)
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
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


def predict(cfg):
    # expdf = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
    expdf = pd.read_csv('./feedback-prize-english-language-learning/test.csv')
    expdf['tokenize_length'] = [len(cfg.tokenizer(text)['input_ids']) for text in expdf['full_text'].values]
    expdf = expdf.sort_values('tokenize_length', ascending=True).reset_index(drop=True)
    
    test_dataset = TestDataset(cfg, expdf)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=cfg.tokenizer, padding='longest'),
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    predictions = []

    weights = glob.glob(CFG.path+"*best.pth")
    for w in weights:
        print(w)
        model = CustomModel(cfg, config_path=cfg.config_path, pretrained=False)
        state = torch.load(w,map_location=torch.device('cpu'))

        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        predictions.append(prediction)
        del model, state; gc.collect()
        
        torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)
    return predictions

class CFG:
    num_workers=4
    path="../input/exp027/"
    config_path=path+'config.pth'
    batch_size=16
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    print(args)

    CFG.path=args.path
    CFG.config_path = CFG.path+'config.pth'
    print(CFG.config_path)
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')

    # test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
    # submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')
    test = pd.read_csv('./feedback-prize-english-language-learning/test.csv')
    # submission = pd.read_csv('./feedback-prize-english-language-learning/sample_submission.csv')
    cfg = CFG()
    predictions = predict(cfg)
    test[CFG.target_cols] = predictions
    print(test.head())