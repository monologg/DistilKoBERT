# DistilKoBERT

Distillation of KoBERT (`SKTBrain KoBERT` 경량화)

**January 27th, 2020 - Update**: 10GB의 Corpus를 가지고 새로 학습하였습니다. Subtask에서 성능이 소폭 상승했습니다.  
**May 14th, 2020 - Update**: 기존 Transformers의 `padding_idx` 이슈를 해결하였습니다. 자세한 사항은 [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)를 참고하시면 됩니다.

## Pretraining DistilKoBERT

- 기존의 12 layer를 **3 layer**로 줄였으며, 기타 configuration은 kobert를 그대로 따랐습니다.
  - [원 논문](https://arxiv.org/abs/1910.01108)은 6 layer를 채택하였습니다.
- Layer 초기화의 경우 기존 KoBERT의 1, 5, 9번째 layer 값을 그대로 사용하였습니다.
- Pretraining Corpus는 한국어 위키, 나무위키, 뉴스 등 약 10GB의 데이터를 사용했으며, 3 epoch 학습하였습니다.

## KoBERT / DistilKoBERT for transformers library

- 기존의 KoBERT를 transformers 라이브러리에서 곧바로 사용할 수 있도록 맞췄습니다.
  - transformers v2.2.2부터 개인이 만든 모델을 transformers를 통해 직접 업로드/다운로드하여 사용할 수 있습니다
  - DistilKoBERT 역시 transformers 라이브러리에서 곧바로 다운 받아서 사용할 수 있습니다.

### Dependencies

- torch==1.1.0
- transformers==2.9.1

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

## Result on Sub-task

|                           | KoBERT      | DistilKoBERT | Bert-multilingual |
| ------------------------- | ----------- | ------------ | ----------------- |
| Model Size (MB)           | 351         | 108          | 681               |
| **NSMC** (acc)            | 89.63       | 88.41        | 87.07             |
| **Naver NER** (F1)        | 86.11       | 84.13        | 84.20             |
| **KorQuAD (Dev)** (EM/F1) | 52.81/80.27 | 54.12/77.80  | 77.04/87.85       |

- NSMC (Naver Sentiment Movie Corpus) ([Implementation of KoBERT-nsmc](https://github.com/monologg/KoBERT-nsmc))
- Naver NER (NER task on Naver NLP Challenge 2018) ([Implementation of KoBERT-NER](https://github.com/monologg/KoBERT-NER))
- KorQuAD (The Korean Question Answering Dataset) ([Implementation of KoBERT-KorQuAD](https://github.com/monologg/KoBERT-KorQuAD))

## Citation

이 코드를 연구용으로 사용하는 경우 아래와 같이 인용해주세요.

```
@misc{park2019distilkobert,
  author = {Park, Jangwon},
  title = {DistilKoBERT: Distillation of KoBERT},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monologg/DistilKoBERT}}
}
```

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [DistilBERT Github](https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [딥러닝으로 동네생활 게시글 필터링하기](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9C%BC%EB%A1%9C-%EB%8F%99%EB%84%A4%EC%83%9D%ED%99%9C-%EA%B2%8C%EC%8B%9C%EA%B8%80-%ED%95%84%ED%84%B0%EB%A7%81%ED%95%98%EA%B8%B0-263cfe4bc58d)
