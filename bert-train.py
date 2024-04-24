import time
# get the start time
st = time.time()

import sys
import os

NODE_NUM = int(sys.argv[1])
GPU_NUM = int(sys.argv[2])
BATCH_SIZE = int(sys.argv[3]) 
NUM_WORKERS = int(sys.argv[4]) 
JOB_NUM = str(sys.argv[5]) 
EPOCHS = int(sys.argv[6]) 

print("Job #" + JOB_NUM + " - training on " + str(NODE_NUM) + " node(s) with " + str(GPU_NUM) + " GPU(s) per node with batch size " + str(BATCH_SIZE) + " and " + str(NUM_WORKERS) + " DataLoader worker(s) for " + str(EPOCHS) + " epochs") 

os.environ['HF_HOME'] = '/raid/michaela/transformers_cache'
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['NCCL_SOCKET_IFNAME'] = 'enp1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import deepspeed
import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoTokenizer, BertTokenizer, BertModel
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import datasets as D
import pandas as pd

torch.cuda.empty_cache()

class seqclassDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = pd.DataFrame(df)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        data_row = self.df.iloc[idx]
        # print(data_row)
        text = str(data_row["text"])
        label = data_row["label"]
        if label == 0:
            label = torch.tensor([1, 0])
        else:
            label = torch.tensor([0, 1])

        tokens = self.tokenizer(text, padding='max_length', return_tensors="pt", max_length=512, truncation=True)
        return dict(input_ids=tokens.input_ids, attn_mask = tokens.attention_mask, labels=label)

# https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
class seqclassDataModule(L.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        
    def setup(self, stage=None):
        self.train_dataset = seqclassDataset(self.train_df, self.tokenizer)
        self.test_dataset = seqclassDataset(self.test_df, self.tokenizer)
        
    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=NUM_WORKERS
        )
    
    def val_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=NUM_WORKERS
        )
    
class BERTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.pre_classifier = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.criterion = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attn_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.criterion(logits.view(-1, 2), labels.float())
        return logits, loss
        
    def training_step(self, batch, batch_idx):
        inputs = batch["input_ids"][0]
        attn_mask = batch["attn_mask"]
        labels = batch["labels"]
        output, loss = self(inputs,attn_mask, labels)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        test_input = batch["input_ids"][0]
        attn_mask = batch["attn_mask"]
        target = batch["labels"]
        
        output, loss = self(test_input,attn_mask, target.float())

        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=0.001)
        return self.optimizer

dataset = D.load_dataset("stanfordnlp/imdb")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data_module = seqclassDataModule(train_df, test_df, tokenizer, BATCH_SIZE)

# # Model
langmodel = BERTModel()

checkpoint_path = "/home/michaela/bert_example/checkpoints/checkpoints" + JOB_NUM

# Trainer
trainer = L.Trainer(
    default_root_dir=checkpoint_path,
    accelerator="gpu",
    num_nodes=NODE_NUM,
    devices=GPU_NUM,
    strategy=DeepSpeedStrategy(
        stage=2
    ),
    precision='16-true',
    max_epochs=EPOCHS,
)

rams_used = []

trainer.fit(langmodel, data_module)

et = time.time()
total_time = et - st
print("EXECUTION TIME (seconds): " + str(total_time))
