from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification, BertForSequenceClassification, LukePreTrainedModel, LukeModel, LukeTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def create_model(args):
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)

    if args.model_name == 'kepler':
        model = MODEL_CLASS.from_pretrained('path/to/KEPLER')
    elif args.model_name == 'luke':
        luke = LukeModel.from_pretrained("studio-ousia/luke-base")
        model = MODEL_CLASS(luke)
    else:
        model = MODEL_CLASS.from_pretrained(args.model_name)
    return model


def get_model_and_tokenizer_class(args):
    if 'gpt' in args.model_name:
        return GPT2LMHeadModel, AutoTokenizer
    elif 'roberta' in args.model_name:
        return RobertaForSequenceClassification, AutoTokenizer
    elif 'kepler' in args.model_name:
        return RobertaForSequenceClassification, AutoTokenizer
    elif 'luke' in args.model_name:
        return LUKEForSequenceClassification, AutoTokenizer
    elif 'bert' in args.model_name:
        return BertForSequenceClassification, AutoTokenizer
    elif 'megatron' in args.model_name:
        return None, AutoTokenizer
    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))

def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'kepler' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'luke' in args.model_name:
        embeddings = model.luke.get_input_embeddings()
    elif 'bert' in args.model_name:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model_name:
        embeddings = model.base_model.get_input_embeddings()
    elif 'megatron' in args.model_name:
        embeddings = model.decoder.embed_tokens
    else:
        raise NotImplementedError()
    return embeddings

class LUKEForSequenceClassification(LukePreTrainedModel):
    def __init__(self, luke):
        super().__init__(luke.config)
        self.num_labels = 2
        self.config = luke.config

        classifier_dropout = self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        self.init_weights()
        self.luke = luke

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )