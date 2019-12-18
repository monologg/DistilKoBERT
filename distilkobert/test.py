import torch
from distilkobert import get_distilkobert_model, get_tokenizer

model = get_distilkobert_model()
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]]).to("cuda")
attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]]).to("cuda")
last_layer_hidden_state, _ = model(input_ids, attention_mask)
print(last_layer_hidden_state)

tokenizer = get_tokenizer()
print(tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]"))
print(tokenizer.convert_tokens_to_ids(['[CLS]', '▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.', '[SEP]']))