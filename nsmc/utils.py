import os
import random
import logging

import torch
import numpy as np
from sklearn.metrics import f1_score

from transformers import BertConfig, DistilBertConfig, BertTokenizer
from tokenization_kobert import KoBertTokenizer

from model import BertClassifier, DistilBertClassifier

MODEL_CLASSES = {
    'kobert': (BertConfig, BertClassifier, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertClassifier, KoBertTokenizer),
    'bert': (BertConfig, BertClassifier, BertTokenizer)
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased'
}


def get_label(args):
    return [0, 1]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='macro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }
