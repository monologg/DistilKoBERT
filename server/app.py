import time
import argparse

import torch
from flask import Flask, jsonify, request

from distilkobert import get_distilkobert_model, get_tokenizer, get_nsmc_model


MODEL_CLASSES = {
    'distilkobert': {
        'model': get_distilkobert_model,
        'tokenizer': get_tokenizer
    },
    'nsmc': {
        'model': get_nsmc_model,
        'tokenizer': get_tokenizer
    }
}


app = Flask(__name__)
tokenizer = None
model = None


def init_model(args):
    global tokenizer, model
    model = MODEL_CLASSES[args.model_type]['model']()
    tokenizer = MODEL_CLASSES[args.model_type]['tokenizer']()


def convert_texts_to_tensors(texts, max_seq_len, add_special_tokens):
    input_ids = []
    attention_mask = []
    for text in texts:
        input_id = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        attention_id = [1] * len(input_id)

        # Zero padding
        padding_length = max_seq_len - len(input_id)
        input_id = input_id + ([tokenizer.pad_token_id] * padding_length)
        attention_id = attention_id + ([0] * padding_length)

        input_ids.append(input_id)
        attention_mask.append(attention_id)

    # Change list to torch tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    return input_ids, attention_mask


@app.route("/predict", methods=["POST"])
def predict():
    rcv_data = request.get_json()
    start_t = time.time()
    # Prediction
    texts = rcv_data['texts']
    max_seq_len = rcv_data['max_seq_len']
    input_ids, attention_mask = convert_texts_to_tensors(texts, max_seq_len, args.add_special_tokens)
    outputs = model(input_ids, attention_mask)
    hidden_state = outputs[0].tolist()

    total_time = time.time() - start_t
    return jsonify(
        output=hidden_state,
        time=total_time
    )


def predict_distilkobert():
    pass


def predict_nsmc():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--port_num", type=int, default=12345, help="Port Number")
    parser.add_argument("-m", "--model_type", type=str, default="distilkobert",
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("-s", "--add_special_tokens", action="store_true", help="Whether to add CLS and SEP token on each texts automatically")
    args = parser.parse_args()

    print("Initializing the {} model...".format(args.model_type))
    init_model(args)

    app.run(host="0.0.0.0", debug=False, port=args.port_num)
