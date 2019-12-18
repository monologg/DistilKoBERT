import os
from .tokenization_kobert import KoBertTokenizer

DOWNLOAD_URL_MAP = {
    'distilkobert': {
        'pytorch_model': ('https://drive.google.com/uc?id=1ysXyKnHWFzX2JAuIEZaXAfH75IhSvMe5', 'pytorch_model.bin'),
        'config': ('https://drive.google.com/uc?id=1XxnqUHQAjKW3AEzDR429a0MzvxBI6qMV', 'config.json')
    },
    'tokenizer': {
        'tokenizer_model': ('https://drive.google.com/uc?id=1_F23OdOyp-uK79bpuUxQsRiV6lt8exBW', 'tokenizer_78b3253a26.model'),
        'vocab': ('https://drive.google.com/uc?id=1Cyty9NbmcOVvDT4JyCliDR0jRZt_j0Or', 'vocab.txt')
    },
    'nsmc': {
        'pytorch_model': ('https://drive.google.com/uc?id=1nmBUpFcWxgTSiVr47VPKSO9YOhuSVw53', 'pytorch_model.bin'),
        'config': ('https://drive.google.com/uc?id=1hq9jEoMIxzyGXsBv7WNk0e4EPH8wwx_M', 'config.json'),
        'training_config': ('https://drive.google.com/uc?id=1VpfLNhg58VtUfl18QxtG4ENDZDKpgRjA', 'training_config.bin')
    }
}


def get_tokenizer():
    return KoBertTokenizer.from_pretrained('monologg/distilkobert')
