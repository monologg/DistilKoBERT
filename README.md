# DistilKoBERT

- Distillation of KoBERT (`SKTBrain KoBERT` 경량화)

## Pretraining DistilKoBERT

- 기존의 12 layer를 **3 layer**로 줄였으며, 기타 configuration은 kobert를 그대로 따랐습니다.
  - [원 논문](https://arxiv.org/abs/1910.01108)은 6 layer를 채택하였습니다.
- Layer 초기화의 경우 기존 KoBERT의 1, 5, 9번째 layer 값을 그대로 사용하였습니다.
- Pretraining Corpus는 한국어 위키, 나무위키, 뉴스 등 약 6GB의 데이터를 사용했으며, 2.5 epoch 학습하였습니다.

## KoBERT / DistilKoBERT for transformers library

- 기존의 KoBERT를 transformers 라이브러리에서 곧바로 사용할 수 있도록 맞췄습니다.
  - transformers v2.2.2부터 개인이 만든 모델을 transformers를 통해 직접 업로드/다운로드하여 사용할 수 있습니다
  - DistilKoBERT 역시 transformers 라이브러리에서 곧바로 다운 받아서 사용할 수 있습니다.

### Dependencies

- torch>=1.1.0
- transformers>=2.2.2
- sentencepiece>=0.1.82

### How to Use

```python
>>> from transformers import BertModel, DistilBertModel
>>> bert_model = BertModel.from_pretrained('monologg/kobert')
>>> distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
```

- Tokenizer를 사용하려면, 루트 디렉토리의 `tokenization_kobert.py` 파일을 복사한 후, `KoBertTokenizer`를 임포트하면 됩니다.
  - KoBERT와 DistilKoBERT 모두 동일한 토크나이저를 사용합니다.
  - **기존 KoBERT의 경우 Special Token이 제대로 분리되지 않는 이슈가 있어서 해당 부분을 수정하여 반영하였습니다.** ([Issue link](https://github.com/SKTBrain/KoBERT/issues/11))

```python
>>> from tokenization_kobert import KoBertTokenizer
>>> tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert도 동일
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## What is different between BERT and DistilBERT

- DistilBert는 기존의 Bert와 달리 token-type embedding을 사용하지 않습니다.

  - Transformers 라이브러리의 DistilBertModel을 사용할 때 기존 BertModel 과 달리 `token_type_ids`를 넣을 필요가 없습니다.

- 또한 DistilBert는 pooler를 사용하지 않습니다.

  - 고로 기존 BertModel의 경우 forward의 return 값으로 `sequence_output, pooled_output, (hidden_states), (attentions)`을 뽑아내지만, DistilBertModel의 경우 `sequence_output, (hidden_states), (attentions)`를 뽑아냅니다.
  - DistilBert에서 `[CLS]` 토큰을 뽑아내려면 `sequence_output[0][:, 0]`를 적용해야 합니다.

```python
# Transformers의 BertModel에서 sequence_output으로부터 pooled_output을 만들기 위해 사용하는 BertPooler
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

## Kobert-Transformers python library

[![Release](https://img.shields.io/badge/release-v0.2.0-green)](https://pypi.org/project/kobert-transformers/)
[![Downloads](https://pepy.tech/badge/kakaotrans)](https://pepy.tech/project/kakaotrans)
[![license](https://img.shields.io/badge/license-Apache%202.0-red)](https://github.com/monologg/DistilKoBERT/blob/master/LICENSE)

- tokenization_kobert.py를 랩핑한 파이썬 라이브러리
- KoBERT, DistilKoBERT를 Huggingface Transformers 라이브러리 형태로 임포트

### Install Kobert-Transformers

```bash
$ pip3 install kobert-transformers
```

### How to Use

```python
>>> import torch
>>> from kobert_transformers import get_distilkobert_model, get_kobert_model

>>> model = get_distilkobert_model()
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> last_layer_hidden_state, _ = model(input_ids, attention_mask)
>>> last_layer_hidden_state
tensor([[[-0.4294,  0.1849,  0.2622,  ..., -0.8856, -0.0617, -0.0664],
         [ 0.0580,  0.2065,  0.1131,  ..., -0.9954, -1.2588, -0.1635],
         [-0.3945,  0.0641, -0.2223,  ..., -0.9819, -0.9723,  0.0929]],

        [[ 0.1698, -0.2389, -0.0153,  ..., -0.0329, -0.0892, -0.0428],
         [ 0.1348, -0.5269, -0.2861,  ..., -0.6471, -0.6776, -0.2948],
         [ 0.0655, -0.4104, -0.0467,  ..., -0.5906, -0.6362, -0.0361]]],
       grad_fn=<AddcmulBackward>)
```

```python
>>> from kobert_transformers import get_tokenizer
>>> tokenizer = get_tokenizer()
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## Result on Sub-task

|                     | KoBERT | DistilKoBERT | Bert-multilingual |
| ------------------- | ------ | ------------ | ----------------- |
| Model Size (MB)     | 351    | 108          | 681               |
| **NSMC** (acc)      | 89.63  | 88.28        | 87.07             |
| **Naver NER** (F1)  | 84.23  | 81.33        | 81.78             |
| **KorQuAD** (EM/F1) | TBD    | TBD          | 77.04/87.85       |

- NSMC (Naver Sentiment Movie Corpus) ([Implementation of KoBERT-nsmc](https://github.com/monologg/KoBERT-nsmc))
- Naver NER (NER task on Naver NLP Challenge 2018) ([Implementation of KoBERT-NER](https://github.com/monologg/KoBERT-NER))

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [DistilBERT Github](https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [딥러닝으로 동네생활 게시글 필터링하기](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9C%BC%EB%A1%9C-%EB%8F%99%EB%84%A4%EC%83%9D%ED%99%9C-%EA%B2%8C%EC%8B%9C%EA%B8%80-%ED%95%84%ED%84%B0%EB%A7%81%ED%95%98%EA%B8%B0-263cfe4bc58d)
