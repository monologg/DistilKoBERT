import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, DistilBertModel, PreTrainedModel, DistilBertConfig
from transformers.modeling_distilbert import DistilBertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BertClassifier(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(BertClassifier, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.label_classifier = FCLayer(bert_config.hidden_size, bert_config.num_labels, args.dropout_rate, use_activation=False)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]

        logits = self.label_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DistilBertClassifier(DistilBertPreTrainedModel):
    def __init__(self, distilbert_config, args):
        super(DistilBertClassifier, self).__init__(distilbert_config)
        self.distilbert = DistilBertModel.from_pretrained(args.model_name_or_path, config=distilbert_config)  # Load pretrained distilbert

        self.num_labels = distilbert_config.num_labels

        self.pre_classifier = nn.Linear(distilbert_config.hidden_size, distilbert_config.hidden_size)
        self.label_classifier = nn.Linear(distilbert_config.hidden_size, distilbert_config.num_labels)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)  # last-layer hidden-state, (all hidden_states), (all attentions)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]  # [CLS]

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.label_classifier(pooled_output)

        outputs = (logits,) + outputs[1:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
