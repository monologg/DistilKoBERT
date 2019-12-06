from distilkobert import get_nsmc_model, get_tokenizer, get_distilkobert_model

model = get_nsmc_model()
del(model)
tokenizer = get_tokenizer()
del(tokenizer)
model = get_distilkobert_model()
del(model)
