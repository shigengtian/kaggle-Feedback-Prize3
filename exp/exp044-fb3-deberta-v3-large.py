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


# # Config

# In[2]:


class CFG:
    str_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train = True
    debug = False
    offline = False
    models_path = 'FB3-models'
    epochs = 5
    save_all_models = False
    competition = 'FB3'
    apex = True
    print_freq = 20
    num_workers = 4
    model = 'microsoft/deberta-v3-large' #If you want to train on the kaggle platform, v3-base is realistic. v3-large will time out.
    loss_func = 'SmoothL1' # 'SmoothL1', 'RMSE'
    gradient_checkpointing = True
    scheduler = 'cosine'
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    #Layer-Wise Learning Rate Decay
    llrd = True
    layerwise_lr = 5e-5
    layerwise_lr_decay = 0.9
    layerwise_weight_decay = 0.01
    layerwise_adam_epsilon = 1e-6
    layerwise_use_bertadam = False
    #pooling
    pooling = 'mean' # mean, max, min, attention, weightedlayer
    layer_start = 4
    #init_weight
    init_weight = 'normal' # normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal
    #re-init
    reinit = True
    reinit_n = 1
    #adversarial
    fgm = False
    awp = False
    adv_lr = 1
    adv_eps = 0.2
    unscale = False
    eps = 1e-6
    betas = (0.9, 0.999)
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    cv_seed = 42
    n_fold = 5
    trn_fold = list(range(n_fold))
    batch_size = 8
    n_targets = 6
    gpu_id = 0
    device = f'cuda:{gpu_id}'
    train_file = './feedback-prize-english-language-learning/train.csv'
    test_file = './feedback-prize-english-language-learning/test.csv'
    submission_file = '../input/feedback-prize-english-language-learning/sample_submission.csv'
    OUTPUT_DIR = "exp044/"


# In[3]:


#Unique model name
if len(CFG.model.split("/")) == 2:
    CFG.identifier = f'{CFG.OUTPUT_DIR}/{CFG.str_now}-{CFG.model.split("/")[1]}'
else:
    CFG.identifier = f'{CFG.OUTPUT_DIR}/{CFG.str_now}-{CFG.model}'


if CFG.train:
    CFG.df_train = pd.read_csv(CFG.train_file)
    # CFG.OUTPUT_DIR = f'./{CFG.identifier}/'
    CFG.log_filename = CFG.OUTPUT_DIR + 'train'
    if CFG.offline:
        #TO DO
        pass
    else:
        os.system('pip install iterative-stratification==0.1.7')
    #CV
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold    
    Fold = MultilabelStratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.cv_seed)
    for n, (train_index, val_index) in enumerate(Fold.split(CFG.df_train, CFG.df_train[CFG.target_cols])):
        CFG.df_train.loc[val_index, 'fold'] = int(n)
    CFG.df_train['fold'] = CFG.df_train['fold'].astype(int)
else:
    #TO DO
    pass

if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]
    if CFG.train:
        CFG.df_train = CFG.df_train.sample(n = 100, random_state = CFG.seed).reset_index(drop=True)
        
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)    
print(CFG.OUTPUT_DIR)


# # Helper function

# In[5]:


def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared = False)
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

def get_logger(filename = CFG.log_filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter('%(message)s'))
    handler2 = FileHandler(filename = f'{filename}.log')
    handler2.setFormatter(Formatter('%(message)s'))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors = None,
        add_special_tokens = True,
        max_length = cfg.max_len,
        pad_to_max_length = True,
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs    

def collate(inputs):
    mask_len = int(inputs['attention_mask'].sum(axis = 1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{int(m)}m {int(s)}s'

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f'{str(asMinutes(s))} (remain {str(asMinutes(rs))})'

def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class RMSELoss(nn.Module):
    def __init__(self, reduction = 'mean', eps = 1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'none')
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
    
seed_everything(CFG.seed)


# # Pooling
# 
# * Attention pooling (https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/361678)
# * WeightedLayerPooling (https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning?scriptVersionId=67176591&cellId=19)

# In[6]:


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings
    
class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim = 1)
        return min_embeddings

#Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

#There may be a bug in my implementation because it does not work well.
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average


# # Fast Gradient Method (FGM)
# Reference :
# 
# https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/143764

# In[7]:


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon = 1., emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


# # Adversarial Weight Perturbation (AWP)
# There may be a bug in my implementation because it does not work well.
# 
# Reference : 
# 
# https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook

# In[8]:


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        
    def attack_backward(self, x, y, attention_mask,epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            with torch.cuda.amp.autocast():
                adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()
            
        self._restore()
        
    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


# # Train function
# * FGM
# * AWP (There may be a bug.)
# * Unscale optimizer

# In[9]:


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    losses = AverageMeter()
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled = CFG.apex)
    start = end = time.time()
    global_step = 0
    if CFG.fgm:
        fgm = FGM(model)
    if CFG.awp:
        awp = AWP(model,
                  optimizer, 
                  adv_lr = CFG.adv_lr, 
                  adv_eps = CFG.adv_eps, 
                  scaler = scaler)
    for step, (inputs, labels) in enumerate(train_loader):
        attention_mask = inputs['attention_mask'].to(device)
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled = CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        if CFG.unscale:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        #Fast Gradient Method (FGM)
        if CFG.fgm:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled = CFG.apex):
                y_preds = model(inputs)
                loss_adv = criterion(y_preds, labels)
                loss_adv.backward()
            fgm.restore()
            
        #Adversarial Weight Perturbation (AWP)
        if CFG.awp:
            loss_awp = awp.attack_backward(inputs, labels, attention_mask, step + 1)
            loss_awp.backward()
            awp._restore()
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f} '
                  'LR: {lr:.8f} '
                  .format(epoch + 1, step, len(train_loader), remain = timeSince(start, float(step + 1)/len(train_loader)),
                          loss = losses,
                          grad_norm = grad_norm,
                          lr = scheduler.get_lr()[0]
                         )
                 )
    return losses.avg


# # Valid function

# In[10]:


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss = losses,
                          remain = timeSince(start, float(step + 1) / len(valid_loader))
                         )
                 )
    return losses.avg, np.concatenate(preds)


# # Logger

# In[11]:


LOGGER = get_logger()
LOGGER.info(f'OUTPUT_DIR: {CFG.OUTPUT_DIR}')


# # Tokenizer

# In[12]:


CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer.save_pretrained(CFG.OUTPUT_DIR + 'tokenizer')

#max_len
lengths = []
tk0 = tqdm(CFG.df_train['full_text'].fillna('').values, total = len(CFG.df_train))
for text in tk0:
    length = len(CFG.tokenizer(text, add_special_tokens = False)['input_ids'])
    lengths.append(length)
CFG.max_len = max(lengths) + 2
LOGGER.info(f'max_len: {CFG.max_len}')


# # Dataset

# In[13]:


class FB3TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype = torch.float)
        return inputs, label


# # Model
# 
# * Initializing module (normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal) 
# * Freeze lower layer when you use large model (v2-xlarge, funnnel, etc.)

# In[14]:


class FB3Model(nn.Module):
    def __init__(self, CFG, config_path = None, pretrained = False):
        super().__init__()
        self.CFG = CFG
        if config_path is None:
            self.config = AutoConfig.from_pretrained(CFG.model, ouput_hidden_states = True)
            self.config.save_pretrained(CFG.OUTPUT_DIR + 'config')
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
            
        LOGGER.info(self.config)
        
        if pretrained:
            self.model = AutoModel.from_pretrained(CFG.model, config=self.config)
            # self.model.save_pretrained(CFG.OUTPUT_DIR + 'model')
        else:
            self.model = AutoModel(self.config)
            
        if self.CFG.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        if CFG.pooling == 'mean':
            self.pool = MeanPooling()
        elif CFG.pooling == 'max':
            self.pool = MaxPooling()
        elif CFG.pooling == 'min':
            self.pool = MinPooling()
        elif CFG.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif CFG.pooling == 'weightedlayer':
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start = CFG.layer_start, layer_weights = None)        
        
        self.fc = nn.Linear(self.config.hidden_size, self.CFG.n_targets)
        self._init_weights(self.fc)
        
        if 'deberta-v2-xxlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False)
        if 'deberta-v2-xlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False)
        if 'funnel-transformer-xlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.blocks[:1].requires_grad_(False)
        if 'funnel-transformer-large' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.blocks[:1].requires_grad_(False)
        if 'deberta-large' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:16].requires_grad_(False)
        if 'deberta-xlarge' in CFG.model:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:36].requires_grad_(False)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
                
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
                
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def feature(self, inputs):
        outputs = self.model(**inputs)
        if CFG.pooling != 'weightedlayer':
            last_hidden_states = outputs[0]
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
        else:
            all_layer_embeddings = outputs[1]
            feature = self.pool(all_layer_embeddings)
            
        return feature
    
    def forward(self, inputs):
        feature = self.feature(inputs)
        outout = self.fc(feature)
        return outout


# # Train
# * Re-initializing upper layer (normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal) 
# * Layer-Wise Learning Rate Dacay (https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning?scriptVersionId=67176591&cellId=29)
# * Loss function, SmoothL1 or RMSE

# In[15]:


def re_initializing_layer(model, config, layer_num):
    for module in model.model.encoder.layer[-layer_num:].modules():
        if isinstance(module, nn.Linear):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data) 
                
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if CFG.init_weight == 'normal':
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif CFG.init_weight == 'xavier_uniform':
                module.weight.data = nn.init.xavier_uniform_(module.weight.data)
            elif CFG.init_weight == 'xavier_normal':
                module.weight.data = nn.init.xavier_normal_(module.weight.data)
            elif CFG.init_weight == 'kaiming_uniform':
                module.weight.data = nn.init.kaiming_uniform_(module.weight.data)
            elif CFG.init_weight == 'kaiming_normal':
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
            elif CFG.init_weight == 'orthogonal':
                module.weight.data = nn.init.orthogonal_(module.weight.data)
                
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return model   

def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    
    train_folds = folds[folds['fold'] != fold].reset_index(drop = True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop = True)
    valid_labels = valid_folds[CFG.target_cols].values
    
    train_dataset = FB3TrainDataset(CFG, train_folds)
    valid_dataset = FB3TrainDataset(CFG, valid_folds)
    
    train_loader = DataLoader(train_dataset,
                              batch_size = CFG.batch_size,
                              shuffle = True, 
                              num_workers = CFG.num_workers,
                              pin_memory = True, 
                              drop_last = True
                             )
    valid_loader = DataLoader(valid_dataset,
                              batch_size = CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers,
                              pin_memory=True, 
                              drop_last=False)

    model = FB3Model(CFG, config_path = None, pretrained = True)
    if CFG.reinit:
        model = re_initializing_layer(model, model.config, CFG.reinit_n)
        
    #os.makedirs(CFG.OUTPUT_DIR + 'config/', exist_ok = True)
    #torch.save(model.config, CFG.OUTPUT_DIR + 'config/config.pth')
    model.to(CFG.device)
    
    def get_optimizer_params(model,
                             encoder_lr,
                             decoder_lr,
                             weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr,
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr,
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr,
             'weight_decay': 0.0}
        ]
        return optimizer_parameters
    
    #llrd
    def get_optimizer_grouped_parameters(model, 
                                         layerwise_lr,
                                         layerwise_weight_decay,
                                         layerwise_lr_decay):
        
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if "model" not in n],
                                         "weight_decay": 0.0,
                                         "lr": layerwise_lr,
                                        },]
        # initialize lrs for every layer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = layerwise_lr
        for layer in layers:
            optimizer_grouped_parameters += [{"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                                              "weight_decay": layerwise_weight_decay,
                                              "lr": lr,
                                             },
                                             {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                                              "weight_decay": 0.0,
                                              "lr": lr,
                                             },]
            lr *= layerwise_lr_decay
        return optimizer_grouped_parameters
    
    if CFG.llrd:
        from transformers import AdamW
        grouped_optimizer_params = get_optimizer_grouped_parameters(model, 
                                                                    CFG.layerwise_lr, 
                                                                    CFG.layerwise_weight_decay, 
                                                                    CFG.layerwise_lr_decay)
        optimizer = AdamW(grouped_optimizer_params,
                          lr = CFG.layerwise_lr,
                          eps = CFG.layerwise_adam_epsilon,
                          correct_bias = not CFG.layerwise_use_bertadam)
    else:
        from torch.optim import AdamW
        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=CFG.encoder_lr, 
                                                    decoder_lr=CFG.decoder_lr,
                                                    weight_decay=CFG.weight_decay)
        optimizer = AdamW(optimizer_parameters, 
                          lr=CFG.encoder_lr,
                          eps=CFG.eps,
                          betas=CFG.betas)
    
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = cfg.num_warmup_steps, 
                num_training_steps = num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps = cfg.num_warmup_steps, 
                num_training_steps = num_train_steps,
                num_cycles = cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    
    if CFG.loss_func == 'SmoothL1':
        criterion = nn.SmoothL1Loss(reduction='mean')
    elif CFG.loss_func == 'RMSE':
        criterion = RMSELoss(reduction='mean')
    
    best_score = np.inf
    best_train_loss = np.inf
    best_val_loss = np.inf
    
    epoch_list = []
    epoch_avg_loss_list = []
    epoch_avg_val_loss_list = []
    epoch_score_list = []
    epoch_scores_list = []

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, CFG.device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time
        
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        
        epoch_list.append(epoch+1)
        epoch_avg_loss_list.append(avg_loss)
        epoch_avg_val_loss_list.append(avg_val_loss)
        epoch_score_list.append(score)
        epoch_scores_list.append(scores)
        
        if best_score > score:
            best_score = score
            best_train_loss = avg_loss
            best_val_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        CFG.OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            
        if CFG.save_all_models:
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        CFG.OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_epoch{epoch + 1}.pth")

    predictions = torch.load(CFG.OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location = torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    
    df_epoch = pd.DataFrame({'epoch' : epoch_list,
                             'MCRMSE' : epoch_score_list,
                             'train_loss' : epoch_avg_loss_list, 
                             'val_loss' : epoch_avg_val_loss_list})
    df_scores = pd.DataFrame(epoch_scores_list)
    df_scores.columns = CFG.target_cols
    
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_train_loss, best_val_loss, valid_folds, pd.concat([df_epoch, df_scores], axis = 1)


# # Run !

# In[16]:


def get_result(oof_df, fold, best_train_loss, best_val_loss):
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
    _output_log = pd.DataFrame([CFG.identifier, CFG.model, CFG.cv_seed, CFG.seed, fold, 'best', score, best_train_loss, best_val_loss] + scores).T
    _output_log.columns = ['file', 'model', 'cv_seed', 'seed', 'fold', 'epoch', 'MCRMSE', 'train_loss', 'val_loss'] + CFG.target_cols
    return _output_log

if CFG.train:
    output_log = pd.DataFrame()
    oof_df = pd.DataFrame()
    train_loss_list = []
    val_loss_list = []
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            best_train_loss, best_val_loss, _oof_df, df_epoch_scores = train_loop(CFG.df_train, fold)
            train_loss_list.append(best_train_loss)
            val_loss_list.append(best_val_loss)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")

            df_epoch_scores['file'] = CFG.identifier
            df_epoch_scores['model'] = CFG.model
            df_epoch_scores['cv_seed'] = CFG.cv_seed
            df_epoch_scores['seed'] = CFG.seed
            df_epoch_scores['fold'] = fold
            df_epoch_scores = df_epoch_scores[['file', 'model', 'cv_seed', 'seed', 'fold', 'epoch', 'MCRMSE', 'train_loss', 'val_loss'] + CFG.target_cols]

            _output_log = get_result(_oof_df, fold, best_train_loss, best_val_loss)
            output_log = pd.concat([output_log, df_epoch_scores, _output_log])

    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    _output_log = get_result(oof_df, 'OOF', np.mean(train_loss_list), np.mean(val_loss_list))
    output_log = pd.concat([output_log, _output_log])
    output_log.to_csv(f'{CFG.identifier}.csv', index=False)
    oof_df.to_pickle(CFG.OUTPUT_DIR+'oof_df.pkl', protocol = 4)

