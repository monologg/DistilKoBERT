# DistilKoBERT

Distillation of KoBERT

## KoBERT for transformers library

- [구글 드라이브 링크](https://drive.google.com/open?id=13jTGc7KrvK9xp9e5GvYjyRz6bf2oJux8)를 통해 KoBERT pretrained model을 다운받을 수 있습니다.

```python
>>> from transformers import BertModel
>>> model = BertModel.from_pretrained('kobert')
```

- Tokenizer를 사용하려면, `kobert` 폴더에서 `tokenization_kobert.py` 파일을 복사한 후, `KoBertTokenizer`를 임포트하면 됩니다.

```python
>>> from tokenization_kobert import KoBertTokenizer
>>> tokenizer = KoBertTokenizer.from_pretrained('kobert')
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## Pretraining DistilKoBERT

- [구글 드라이브 링크](https://drive.google.com/open?id=15Ro2LKfXEtdGdRTx15iraxREiFnD5Zig)를 통해 DistilKoBERT pretrained model을 다운받을 수 있습니다.
- 기존의 12 layer를 3 layer로 줄였으며, 기타 configuration은 kobert를 그대로 따랐습니다.
- Pretraining Corpus는 위키, 나무위키, 뉴스 등 약 3GB의 데이터를 사용했으며, 2 epoch 학습하였습니다. (추후 더 많은 데이터로 학습 예정)

## Install DistilKoBERT

```bash
$ pip3 install distilkobert
```

## How to use

```python
>>> import torch
>>> from distilkobert import get_distilkobert_model

>>> model = get_distilkobert_model()
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> last_layer_hidden_state, _ = model(input_ids, attention_mask)
>>> last_layer_hidden_state
tensor([[[-0.2155,  0.1182,  0.1865,  ..., -1.0626, -0.0747, -0.0945],
         [-0.5559, -0.1476,  0.1060,  ..., -0.3178, -0.0172, -0.1064],
         [ 0.1284,  0.2212,  0.2971,  ..., -0.4619,  0.0483,  0.3293]],

        [[ 0.0414, -0.2016,  0.2643,  ..., -0.4734, -0.9823, -0.2869],
         [ 0.2286, -0.1787,  0.1831,  ..., -0.7605, -1.0209, -0.5340],
         [ 0.2507, -0.0022,  0.4103,  ..., -0.7278, -0.9471, -0.3140]]],
       grad_fn=<AddcmulBackward>)
```

```python
>>> from distilkobert import get_tokenizer
>>> tokenizer = get_tokenizer()
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## Result

|                 | KoBERT | DistilKoBERT (3 layer) | Bert-base-multilingual-cased | FastText |
| --------------- | ------ | ---------------------- | ---------------------------- | -------- |
| Model Size (MB) | 351    | 108                    | 681                          | 2        |
| NSMC (%)        | 89.63  | 87.71                  | 87.07                        | 85.50    |

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [DistilBERT](https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
