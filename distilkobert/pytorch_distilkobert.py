import os

from transformers import DistilBertModel, DistilBertConfig

from distilkobert.utils import DOWNLOAD_URL_MAP, download


def get_distilkobert_model(cachedir='~/distilkobert/'):
    f_cachedir = os.path.expanduser(cachedir)
    download(DOWNLOAD_URL_MAP['pytorch_model'][0], DOWNLOAD_URL_MAP['pytorch_model'][1])
    download(DOWNLOAD_URL_MAP['config'][0], DOWNLOAD_URL_MAP['config'][1])

    config = DistilBertConfig.from_pretrained(f_cachedir)
    model = DistilBertModel.from_pretrained(f_cachedir, config=config)  # Load pretrained distilbert

    return model
