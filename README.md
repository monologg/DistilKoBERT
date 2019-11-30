# DistilKoBERT

Distillation of KoBERT

## KoBERT for transformers library

- [구글 드라이브 링크](https://drive.google.com/open?id=13jTGc7KrvK9xp9e5GvYjyRz6bf2oJux8)를 통해 KoBERT pretrained model을 다운받을 수 있습니다.
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

TBD

## Result

|      | KoBERT | DistilKoBERT |
| ---- | ------ | ------------ |
| NSMC | 89.628 | TBD          |

## Reference

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [DistilBERT](https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md)
