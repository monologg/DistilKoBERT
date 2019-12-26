from transformers import DistilBertModel, BertModel


def get_distilkobert_model():
    model = DistilBertModel.from_pretrained('monologg/distilkobert')
    return model


def get_kobert_model():
    model = BertModel.from_pretrained('monologg/kobert')
    return model
