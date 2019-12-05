import os

import torch
from transformers import DistilBertModel, DistilBertConfig

from .utils import DOWNLOAD_URL_MAP, download
from .nsmc_model import DistilBertClassifier


def get_distilkobert_model(cachedir='~/distilkobert/', no_cuda=False):
    f_cachedir = os.path.expanduser(cachedir)
    download(DOWNLOAD_URL_MAP['distilkobert']['pytorch_model'][0], DOWNLOAD_URL_MAP['distilkobert']['pytorch_model'][1], cachedir)
    download(DOWNLOAD_URL_MAP['distilkobert']['config'][0], DOWNLOAD_URL_MAP['distilkobert']['config'][1], cachedir)

    config = DistilBertConfig.from_pretrained(f_cachedir)
    model = DistilBertModel.from_pretrained(f_cachedir, config=config)  # Load pretrained distilbert

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    model.to(device)

    return model


def get_nsmc_model(cachedir='~/nsmc/', no_cuda=False):
    f_cachedir = os.path.expanduser(cachedir)
    download(DOWNLOAD_URL_MAP['nsmc']['pytorch_model'][0], DOWNLOAD_URL_MAP['nsmc']['pytorch_model'][1], cachedir)
    download(DOWNLOAD_URL_MAP['nsmc']['config'][0], DOWNLOAD_URL_MAP['nsmc']['config'][1], cachedir)
    download(DOWNLOAD_URL_MAP['nsmc']['training_config'][0], DOWNLOAD_URL_MAP['nsmc']['training_config'][1], cachedir)

    bert_config = DistilBertConfig.from_pretrained(f_cachedir)
    model = DistilBertClassifier.from_pretrained(f_cachedir, config=bert_config)

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    model.to(device)

    return model
