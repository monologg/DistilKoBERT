import os

import torch
from transformers import DistilBertModel, DistilBertConfig


def get_distilkobert_model(no_cuda=False):
    config = DistilBertConfig.from_pretrained('monologg/distilkobert')
    model = DistilBertModel.from_pretrained('monologg/distilkobert', config=config)  # Load pretrained distilbert

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    model.to(device)

    return model
