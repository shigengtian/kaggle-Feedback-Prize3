import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer
from transformers import TrainingArguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %env TOKENIZERS_PARALLELISM=false
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CFG:
    DEBUG = False 
    EXP_NUM = 1
    MODEL_NAME = "microsoft/deberta-v3-base"
    
    BS = 1 if DEBUG else 4
    N_EPOCHS = 1 if DEBUG else 5
    GRAD_ACCUM = 1 if DEBUG else 2
    LR = 2e-5
    
    SEED = 0
    TEST_SIZE = 0.2


train_df = pd.read_csv("./feedback-prize-english-language-learning/train.csv")[['text_id', 'full_text']]
train_df_2021 = pd.read_csv("./feedback-prize-english-language-learning/2021_pseduo_label.csv")

print(len(train_df))

# train_df = pd.concat([train_df, train_df_2021])
# print(len(train_df))


texts = train_df["full_text"].tolist()
if CFG.DEBUG:
    texts = texts[:100]

train_text_list, valid_text_list, _, _ = train_test_split(texts, texts, test_size=CFG.TEST_SIZE, random_state=CFG.SEED)
print(len(train_text_list))

mlm_train_json_path = f'train_mlm.json'
mlm_valid_json_path = f'valid_mlm.json'


for json_path, list_ in zip([mlm_train_json_path, mlm_valid_json_path],
                            [train_text_list, valid_text_list]):
    with open(str(json_path), 'w') as f:
        for sentence in list_:
            row_json = {'text': sentence}
            json.dump(row_json, f)
            f.write('\n')

datasets = load_dataset(
    'json',
    data_files={'train': str(mlm_train_json_path),
                'valid': str(mlm_valid_json_path)},
    )

# print(datasets["train"][:2])


tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, trim_offsets=False)
def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=1,
    remove_columns=["text"],
    batch_size=CFG.BS)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


config = AutoConfig.from_pretrained(CFG.MODEL_NAME, output_hidden_states=True)
model = AutoModelForMaskedLM.from_pretrained(CFG.MODEL_NAME, config=config)

ds_config_dict = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

training_args = TrainingArguments(
    output_dir="deberta-base_2022",
    evaluation_strategy="epoch",
    learning_rate=CFG.LR,
    weight_decay=0.01,
    save_strategy='epoch',
    per_device_train_batch_size=CFG.BS,
    num_train_epochs=CFG.N_EPOCHS,
    run_name=f'deberta-base-2022-{CFG.EXP_NUM}',
    logging_dir='./logs',
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=200,
    gradient_checkpointing=True,
    gradient_accumulation_steps=CFG.GRAD_ACCUM,
    deepspeed=ds_config_dict,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets['valid'],
    data_collator=data_collator,
)

trainer.train()