import torch
from transformers import DistilBertModel, BertModel


def get_distilkobert_model(no_cuda=False):
    model = DistilBertModel.from_pretrained('monologg/distilkobert')

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    model.to(device)

    return model


def get_kobert_model(no_cuda=False):
    model = BertModel.from_pretrained('monologg/kobert')

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    model.to(device)

    return model
