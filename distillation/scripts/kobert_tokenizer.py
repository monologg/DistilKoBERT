import sentencepiece as spm
from kobert.utils import get_tokenizer
# from transformers import BertTokenizer


class KoBertTokenizer(object):
    """
    https://github.com/google/sentencepiece/blob/master/python/README.md
    """

    def __init__(self, cache_dir=None):
        self.sp = spm.SentencePieceProcessor()
        if cache_dir:
            self.sp.Load(cache_dir)
        else:
            self.sp.Load(get_tokenizer())

    def __len__(self):
        return 8002

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def decode(self, token_ids):
        return self.sp.DecodeIds(token_ids)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.sp.PieceToId(token))
        return ids

    def encode_id(self, piece):
        return self.sp.PieceToId(piece)

if __name__ == "__main__":
    tokenizer = KoBertTokenizer()

    text = "[CLS] 한국어 모델이당~~ kor [SEP]"
    a = tokenizer.tokenize(text)
    print(a)
    print(tokenizer.convert_tokens_to_ids(a))
    a = tokenizer.encode(text)
    print(a)

    print(len(tokenizer))