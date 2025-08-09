# %%
import os
import torch.nn.functional as F
import datasets
import collections
import random
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, adjusted_rand_score, \
    normalized_mutual_info_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import transformers
from collections.abc import Mapping
import shutil
from torchgen.api.cpp import return_type
from transformers import Trainer, PreTrainedModel, BertConfig, PretrainedConfig
from transformers.training_args import TrainingArguments
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import roc_auc_score
from performer_pytorch import Performer
import torch
import torch.nn as nn
import dgl
import pickle
import datetime


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    dgl.seed(seed)


def create_path(output_dir, fold=None):
    if fold:
        output_dir = output_dir + f"fold{fold}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


class ModelOutput(Mapping):
    def __init__(self, logit=None, labels=None, loss=None, cell_name=None, attention=None):
        self.logit = logit
        self.labels = labels
        self.loss = loss
        self._data = {"loss": loss, "logit": logit, "labels": labels}
        self.cell_name = cell_name
        self.attention = attention

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[list(self._data.keys())[key]]
        if isinstance(key, slice):
            return [self._data[k] for k in list(self._data.keys())[key]]
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __setitem__(self, key, value):
        self._data[key] = value
        setattr(self, key, value)

    def key(self):
        return self._data.keys()


def compute_metrics(pred):
    logits, labels = pred
    logits = logits[0]
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)

    roc_auc = roc_auc_score(labels, logits[:, 1])

    f1 = f1_score(labels, preds, average='weighted')
    aupr = average_precision_score(labels, logits[:, 1])
    # ari = adjusted_rand_score(labels, preds)
    # ami = normalized_mutual_info_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return {
        "acc": acc,
        'roc': roc_auc,
        "f1": f1,
        "aupr": aupr,

        "mcc": mcc,
    }


class CustomDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [torch.tensor(example["input_ids"]) for example in batch]
        labels = [torch.tensor(example["label"]) for example in batch]
        numeric_ids = [torch.tensor(example["numeric_ids"], dtype=torch.int) for example in batch]
        cell_names = [example["cell_name"] for example in batch]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        numeric_ids_padded = pad_sequence(numeric_ids, batch_first=True, padding_value=0)

        labels = torch.tensor(labels)
        attention_mask = (input_ids_padded != self.pad_token_id).float()

        return {
            "input_ids": input_ids_padded,
            "labels": labels,
            "numeric_ids": numeric_ids_padded,
            "attention_mask": attention_mask,
            "cell_name": cell_names
        }


    # data_collator = CustomDataCollator(pad_token_id=token_dictionary["<pad>"])


class CustomTrainer(Trainer):
    def predict(self, test_dataset, *args, **kwargs):
        output = super().predict(test_dataset, *args, **kwargs)
        additional_info = {
            'cellname': test_dataset["cell_name"],
        }

        return output, additional_info

    def calculate_custom_metric(self, predictions, labels):
        preds = predictions.argmax(axis=1)
        accuracy = (preds == labels).mean()
        return accuracy


class MyCustomConfig(PretrainedConfig):
    def __init__(self, input_dim=256, hidden_dim=128, num_exper=5000, num_genes=10000, num_layers=1, num_heads=1):
        super(MyCustomConfig, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_exper = num_exper
        self.num_genes = num_genes
        self.num_layers = num_layers
        self.num_heads = num_heads
