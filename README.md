# DistilKoBERT

Distillation of KoBERT

## KoBERT for transformers library

- Download the KoBERT pretrained model from this [Google Drive link](https://drive.google.com/open?id=13jTGc7KrvK9xp9e5GvYjyRz6bf2oJux8).
- For using tokenizer, copy `tokenization_kobert.py` from `kobert` folder, and import `KoBERTTokenizer`.

```python
>>> from tokenization_kobert import KoBertTokenizer
>>> tokenizer = KoBertTokenizer.from_pretrained('kobert')
>>> tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]")
['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]'])
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]
```

## Pretraining DistilKoBERT

TBD

## Result

|      | KoBERT | DistilKoBERT |
| ---- | ------ | ------------ |
| NSMC | 89.628 | TBD          |

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [DistilBERT](https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md)
