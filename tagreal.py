import os
import sys
from datetime import datetime 
import random
from copy import deepcopy
from eval_utils import *
import argparse
import torch
from torch.utils.data import Dataset
import json
from os.path import join, abspath, dirname
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers import LukeTokenizer
from transformers import BertPreTrainedModel, AutoModel, PreTrainedModel
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification, BertForSequenceClassification, LukePreTrainedModel, LukeModel, LukeTokenizer, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import distributed as dist
import numpy as np
from tqdm import tqdm

SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased', 'bert-base-uncased', 'bert-large-uncased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large', 'luke', 'kepler',
                  'megatron_11b', 'biobert', 'sapbert']

# logger_set = setup_logger(name='set_logger', f_name='FB60K_NYT_set')


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def construct_generation_args():

    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--model_name", type=str, default='roberta-large', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--template", type=str, default="(5, 5, 5)")
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--valid_step", type=int, default=10000)
    parser.add_argument("--recall_k", type=int, default=64)
    parser.add_argument("--pos_K", type=int, default=1)
    parser.add_argument("--neg_K", type=int, default=1)
    parser.add_argument("--random_neg_ratio", type=float, default=1.0)
    parser.add_argument("--kge_neg", type=str, default='all', choices=['all', 'tail'])

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lm_lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # lama configuration
    parser.add_argument("--use_original_template", action='store_true')
    parser.add_argument("--use_lm_finetune", action='store_true')

    parser.add_argument("--link_prediction", action='store_true')
    parser.add_argument("--output_cla_results", action='store_true')
    parser.add_argument("--add_definition", action='store_true')
    parser.add_argument("--test_open", action='store_true')

    # support info
    parser.add_argument("--supp_info", type=str, default="False")

    # directories
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--out_dir", type=str, default='./dataset')
    parser.add_argument("--load_dir", type=str, default='')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args


def create_model(args):
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)

    if args.model_name == 'kepler':
        model = MODEL_CLASS.from_pretrained('path/to/KEPLER')
    elif args.model_name == 'luke':
        luke = LukeModel.from_pretrained("studio-ousia/luke-base")
        model = MODEL_CLASS(luke)
    elif args.model_name == 'biobert':
        model = MODEL_CLASS.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    elif args.model_name == 'sapbert':
        sapbert = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = MODEL_CLASS(sapbert)
    else:
        model = MODEL_CLASS.from_pretrained(args.model_name)
    return model


def get_model_and_tokenizer_class(args):
    if 'gpt' == args.model_name:
        return GPT2LMHeadModel, AutoTokenizer
    elif 'roberta' == args.model_name:
        return RobertaForSequenceClassification, AutoTokenizer
    elif 'kepler' == args.model_name:
        return RobertaForSequenceClassification, AutoTokenizer
    elif 'luke' == args.model_name:
        return LUKEForSequenceClassification, AutoTokenizer
    elif 'bert' == args.model_name[:4]:
        return BertForSequenceClassification, AutoTokenizer
    elif 'megatron' == args.model_name:
        return None, AutoTokenizer
    elif 'biobert' == args.model_name:
        return AutoModelForSequenceClassification, AutoTokenizer
    elif 'sapbert' == args.model_name:
        return SapBertForSequenceClassfication, AutoTokenizer

    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


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
                    # print("single_label_classification")
                    self.config.problem_type = "single_label_classification"
                else:
                    # print("multi_label_classification")
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


class SapBertForSequenceClassfication(BertPreTrainedModel):
    def __init__(self, sapbert):
        super().__init__(sapbert.config)
        self.num_labels = 2
        self.config = sapbert.config

        classifier_dropout = self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, 2)

        self.init_weights()
        self.sapbert = sapbert

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.sapbert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # labels=labels
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
                    # print("single_label_classification")
                    self.config.problem_type = "single_label_classification"
                else:
                    # print("multi_label_classification")
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
    


def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'kepler' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'luke' in args.model_name:
        embeddings = model.luke.get_input_embeddings()
    elif 'bert' == args.model_name[:4]:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model_name:
        embeddings = model.base_model.get_input_embeddings()
    elif 'sapbert' in args.model_name:
        embeddings = model.sapbert.get_input_embeddings()
    elif 'megatron' in args.model_name:
        embeddings = model.decoder.embed_tokens
    else:
        raise NotImplementedError()
    return embeddings


class KEPromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args, relation_num):
        super().__init__()
        self.device = device
        self.template = template
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        self.relation_num = relation_num
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.spell_length
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices_relation = torch.LongTensor(list(range(sum(self.template)))).to(self.device)
        # embedding
        self.embedding_relation = torch.nn.Embedding(sum(self.template) * self.relation_num, self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self, rs_tensor):
        if sum(self.template) == 0:
            return None

        # bz x template
        seq_indices_relation_spec = self.seq_indices_relation.unsqueeze(0) + rs_tensor.unsqueeze(-1) * sum(self.template)
        # bz x template x dim
        input_embeds = self.embedding_relation(seq_indices_relation_spec)
        
        # output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        # return output_embeds

        return input_embeds

    def get_query(self, texts, rs, prompt_tokens):
        contents = texts.split('\t\t')
        ans_list = [self.tokenizer.cls_token_id]

        # length = 5 for triple setting
        for i in range(len(contents)):
            ans_list += prompt_tokens * self.template[i]
            if len(contents[i]) != 0:
                if len(ans_list) == 1:
                    ans_list += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(contents[i]))
                else:
                    ans_list += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + contents[i]))
        ans_list += prompt_tokens * self.template[-1]
        ans_list += [self.tokenizer.sep_token_id]

        return [ans_list]


def get_vocab_by_strategy(args, tokenizer):
    return tokenizer.get_vocab()


class BasicDataWiki:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = args.data_dir
        self.tokenizer = tokenizer
        self.triple2text = None
        self.query2text_tail = None
        self.query2text_head = None
        self.init_templates()
        self.init_definition()

    def init_definition(self):
        if os.path.exists(os.path.join(self.dataset, 'triple2text.txt')):
            self.triple2text = {}
            lines = open(os.path.join(self.dataset, 'triple2text.txt'))
            print('loading bm25 triple2text ...')
            for line in tqdm(lines):
                triple, text = line.split('####SPLIT####')
                h, r, t = triple.split('||')
                triple_ = h +'\t' + r + '\t' + t
                self.triple2text[triple_] = text[:-1]
            # logger_set.info(f'triple2text{self.triple2text}')
            
        else:
            print('no bm25 triple2text found .')
            self.triple2text = None

        if os.path.exists(os.path.join(self.dataset, 'query2text_tail.txt')):
            self.query2text_tail = {}
            lines = open(os.path.join(self.dataset, 'query2text_tail.txt'))
            print('loading bm25 query2text_tail ...')
            for line in tqdm(lines):
                query, text = line.split('####SPLIT####')
                h, r = query.split('||')
                query_ = h +'||' + r
                self.query2text_tail[query_] = text[:-1]
            
        else:
            print('no bm25 query2text_tail found')
            self.query2text_tail = None

        if os.path.exists(os.path.join(self.dataset, 'query2text_head.txt')):
            self.query2text_head = {}
            lines = open(os.path.join(self.dataset, 'query2text_head.txt'))
            print('loading bm25 query2text_head ...')
            for line in tqdm(lines):
                query, text = line.split('####SPLIT####')
                t, r = query.split('||')
                query_ = t +'||' + r
                self.query2text_head[query_] = text[:-1]
            
        else:
            print('no bm25 query2text_head found')
            self.query2text_head = None

    def init_templates_others(self):
        with open(f'{self.dataset}/relation2template.json', 'r') as f:
            self.relation2template = json.load(f)
        self.entity2label = None
        
    def init_templates(self):
        entity_label_file = f'{self.dataset}/entity2label.txt'
        relation_label_file = f'{self.dataset}/relation2label.json'
        
        train_file = f'{self.dataset}/train.txt'
        valid_file = f'{self.dataset}/valid.txt'
        test_file = f'{self.dataset}/test.txt'

        self.init_templates_others()

        self.entity2label = {}
        self.relation2label = None
        lines = open(entity_label_file).readlines()
        for line in lines:
            entity, label = line.strip().split('\t')
            self.entity2label[entity] = label

        r_list = list(set([line.strip().split('\t')[1] for line in open(train_file).readlines() + open(valid_file).readlines() + open(test_file).readlines()]))
        self.relation2idx = {r_list[i]: i for i in range(len(r_list))}


class BasicDatasetWiki(Dataset):
    def __init__(self, basic_data):
        super().__init__()
        self.basic_data = basic_data
        self.relation2label = basic_data.relation2label
        self.relation2template = basic_data.relation2template
        self.entity2label = basic_data.entity2label
        self.relation2idx = basic_data.relation2idx
        self.triple2text = basic_data.triple2text
        self.query2text_tail = basic_data.query2text_tail
        self.query2text_head = basic_data.query2text_head
        self.triples = None

    def convert_from_triple_to_sentence(self, triple, isTrain=False, reverse=False, cnt=[]):
        h, r, t = triple
        triple_ = h + '\t' + r + '\t' + t
        query_ = h + '||' + r

        h_, t_ = self.entity2label[h], self.entity2label[t]

        this_template = self.relation2template[r].strip()

        # if training phase, convert triple to text, else, convert query to text
        if isTrain:
            if self.triple2text is not None:
                sentence = self.triple2text[triple_] if triple_ in self.triple2text.keys() else ''
                sentence = sentence if len(sentence) <= 100 else ''
                if sentence != '':
                    cnt.append(1)
                sentence = ''
                this_template = f'{sentence} [SEP] {this_template}'
            else:
                this_template = f'[SEP] {this_template}'
        else:
            if reverse == False and self.query2text_tail is not None:
                sentence = self.query2text_tail[query_] if query_ in self.query2text_tail.keys() else ''
                # sentence = sentence if len(sentence) <= 100 else ''
                if sentence != '':
                    cnt.append(1)
                this_template = f'{sentence} [SEP] {this_template}'

            if reverse == True and self.query2text_head is not None:
                sentence = self.query2text_head[query_] if query_ in self.query2text_head.keys() else ''
                # sentence = sentence if len(sentence) <= 100 else ''
                if sentence != '':
                    cnt.append(1)
                this_template = f'{sentence} [SEP] {this_template}'


        if self.entity2label is not None:
            h, t = self.entity2label[h], self.entity2label[t]

        this_template = this_template.replace('[X]', '::;;##').replace('[Y]', '::;;##')
        prompts = this_template.split('::;;##')
        prompts = [x.strip() for x in prompts]
        assert(len(prompts) == 3)

        idx_x = self.relation2template[r].find('[X]')
        idx_y = self.relation2template[r].find('[Y]')
        if idx_x < idx_y:
            final_list = [prompts[0], h.strip().replace('_', ' '), prompts[1], t.strip().replace('_', ' '), prompts[2]]
        else:
            final_list = [prompts[0], t.strip().replace('_', ' '), prompts[1], h.strip().replace('_', ' '), prompts[2]]
        return '\t\t'.join(final_list)

    def __getitem__(self, i):
        if self.triples is None:
            return self.texts[i], self.rs[i], self.labels[i]
        else:
            return self.texts[i], self.rs[i], self.labels[i], self.triples[i]

    def __len__(self):
        return len(self.labels)


class KEDatasetWiki(BasicDatasetWiki):
    def __init__(self, pos_file, neg_file_random, basic_data, neg_file_kge=None, pos_K=1, neg_K=1, random_neg_ratio=1.0, isTrain=True):
        super().__init__(basic_data)
        self.pos_K = pos_K
        self.neg_K = neg_K
        self.isTrain = isTrain
        self.random_neg_ratio = random_neg_ratio
        self.texts, self.rs, self.labels, self.triples = self.process_data(pos_file, neg_file_random, neg_file_kge)

    def process_data(self, pos_file, neg_file_random, neg_file_kge):
        relation_list = []
        texts, rs, labels, triples = [], [], [], []
        pos_lines = open(pos_file).readlines()
        neg_rand_lines = open(neg_file_random).readlines()[:-1]
        if neg_file_kge is not None:
            neg_kge_lines = open(neg_file_kge).readlines()[:-1]
            # random.shuffle(neg_kge_lines)
        # WARNING: data must be shuffled

        random.shuffle(neg_rand_lines)
        rand_neg_k = int(self.neg_K * self.random_neg_ratio)
        kge_neg_k = self.neg_K - rand_neg_k
        count = []
        for i in range(len(pos_lines)):
            pos_triple = pos_lines[i].strip().split('\t')
            for x in range(self.pos_K):
                texts.append(self.convert_from_triple_to_sentence(pos_triple, self.isTrain, cnt=count))
                labels.append(1)
                rs.append(self.relation2idx[pos_triple[1]])
                triples.append('\t'.join(pos_triple))
            for x in range(rand_neg_k * i, rand_neg_k * (i + 1)): 
                neg_triple = neg_rand_lines[x].strip().split('\t')
                texts.append(self.convert_from_triple_to_sentence(neg_triple, self.isTrain, cnt=count))
                labels.append(0)
                rs.append(self.relation2idx[neg_triple[1]])
                triples.append('\t'.join(neg_triple))
            for x in range(kge_neg_k * i, kge_neg_k * (i + 1)):
                neg_triple = neg_kge_lines[x].strip().split('\t')
                texts.append(self.convert_from_triple_to_sentence(neg_triple, self.isTrain, cnt=count))
                labels.append(0)
                rs.append(self.relation2idx[neg_triple[1]])
                triples.append('\t'.join(neg_triple))
        print(f'classification support info count: {len(count)}')
        return texts, rs, labels, triples


class KEDatasetWikiInfer(BasicDatasetWiki):
    def __init__(self, filename, basic_data, recall_k, head=False):
        super().__init__(basic_data)
        self.get_lines(filename, recall_k)
        self.texts, self.rs, self.labels = self.process_data(filename, head)

    def get_lines(self, filename, recall_k):
        lines = open(filename).read()
        triples = lines.strip().split('SPLIT\n')
        triple_set = set()
        for index, triple in enumerate(triples):
            lines = triple.strip().split('\n')
            if index == len(triples) - 1:
                lines = lines[:-1]
            for i in range(min(recall_k, len(lines))):
                triple_set.add(lines[i].strip())
        self.triple_list = list(triple_set)

    def process_data(self, filename, head=False):
        count = []
        texts, rs, labels = [], [], []
        for i in tqdm(range(len(self.triple_list))):
            pos_triple = self.triple_list[i].strip().split('\t')
            if head == False:
                texts.append(self.convert_from_triple_to_sentence(pos_triple, cnt=count))
            else:
                texts.append(self.convert_from_triple_to_sentence(pos_triple, reverse=True, cnt=count))
            labels.append(1)
            rs.append(self.relation2idx[pos_triple[1]])
        print(f'link support info count: {len(count)}')
        return texts, rs, labels


def get_dataloader(args, tokenizer):
    basic_data = BasicDataWiki(args, tokenizer)

    neg_file_kge = join(args.data_dir, f'train_neg_kge_{args.kge_neg}.txt')
    if args.random_neg_ratio == 1.0:
        neg_file_kge = None

    train_set = KEDatasetWiki(
        join(args.data_dir, 'train.txt'), 
        join(args.data_dir, 'train_neg_rand.txt'),
        basic_data,
        neg_file_kge=neg_file_kge,
        pos_K=args.pos_K,
        neg_K=args.neg_K,
        random_neg_ratio=args.random_neg_ratio,
        isTrain=True
    )
    print("Finished building train set")

    test_set = KEDatasetWiki(
        join(args.data_dir, 'test_pos.txt'), 
        join(args.data_dir, 'test_neg.txt'), 
        basic_data,
        isTrain=True
    )
    print("Finished building test set")

    dev_set = KEDatasetWiki(
        join(args.data_dir, 'valid_pos.txt'), 
        join(args.data_dir, 'valid_neg.txt'), 
        basic_data,
        isTrain=True
    )
    print("Finished building valid set")

    # if args.test_open:
    #     o_test_set = KEDatasetWiki(
    #         join(args.data_dir, 'o_test_pos.txt'), 
    #         join(args.data_dir, 'o_test_neg.txt'), 
    #         basic_data
    #     )
    if args.link_prediction:
        link_dataset_tail = KEDatasetWikiInfer(
            join(args.data_dir, 'link_prediction_tail.txt'), 
            basic_data, 
            args.recall_k
        )
        link_dataset_head = KEDatasetWikiInfer(
            join(args.data_dir, 'link_prediction_head.txt'), 
            basic_data, 
            args.recall_k,
            head=True
        )
    
    # tail_set_text = link_dataset_tail.texts
    # out_file = open('./tttt.txt', 'w', encoding='utf-8')
    # for text in tqdm(tail_set_text):
    #     out_file.write(text + '\n')
    # out_file.close()

    # head_set_text = link_dataset_head.texts
    # out_file = open('./hhhh.txt', 'w', encoding='utf-8')
    # for text in tqdm(head_set_text):
    #     out_file.write(text + '\n')
    # out_file.close()

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # dev_loader = DataLoader(dev_set, batch_size=args.batch_size)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    ch_test_loader, oh_test_loader = None, None

    # if args.test_open:
    #     o_test_loader = DataLoader(o_test_set, batch_size=args.batch_size)
    # else:
    o_test_loader = None

    # if args.link_prediction:
    #     # link_loader_tail = DataLoader(link_dataset_tail, batch_size=args.batch_size)
    #     # link_loader_head = DataLoader(link_dataset_head, batch_size=args.batch_size)
    # else:
    #     link_loader_tail = None
    #     link_loader_head = None
    #     link_dataset_tail = None
    #     link_dataset_head = None
    return train_set, test_set, dev_set, link_dataset_head, link_dataset_tail, ch_test_loader, oh_test_loader, o_test_loader, len(basic_data.relation2idx)


class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, device_ids, template, tokenizer_src, relation_num):
        super().__init__()
        self.args = args
        self.device = device
        self.relation_num = relation_num
        self.tokenizer = tokenizer_src
        self.template = template
        self.device_ids = device_ids

        self.model = create_model(self.args)
        
        ##modified
        print(torch.cuda.device_count())  # 打印gpu数量
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        print('world_size', torch.distributed.get_world_size()) # 打印当前进程数
        torch.cuda.set_device(self.args.local_rank)
        
        
        if len(self.device_ids) > 1:
            self.model = self.model.cuda(args.local_rank)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=False, 
                                                     broadcast_buffers=False)
            
        # self.model.module.to(self.device)
        # self.model.to(self.device)
        
        
        
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
            
        ## modified
        self.embeddings = get_embedding_layer(self.args, self.model.module)
        # self.embeddings = get_embedding_layer(self.args, self.model)


        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = KEPromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args, self.relation_num)
        self.prompt_encoder = self.prompt_encoder.to(self.device)


    def embed_input(self, queries, rs_tensor):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder(rs_tensor)
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]
        return raw_embeds

    def forward_classification(self, texts, rs, labels, return_candidates=False, bz=None):
        if self.args.model_name == 'luke':
            return self.forward_classification_luke(texts, rs, labels, return_candidates, bz)
        bz = len(texts)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        queries = [torch.LongTensor(self.prompt_encoder.get_query(texts[i], rs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().cuda(self.args.local_rank, non_blocking=True)

        # construct label ids
        attention_mask = queries != self.pad_token_id
        rs_tensor = torch.LongTensor(rs).cuda(self.args.local_rank, non_blocking=True)

        # get embedded input
        inputs_embeds = self.embed_input(queries, rs_tensor)

        output = self.model(inputs_embeds=inputs_embeds.cuda(self.args.local_rank, non_blocking=True),
                            attention_mask=attention_mask.cuda(self.args.local_rank, non_blocking=True).bool(),
                            labels=labels.cuda(self.args.local_rank, non_blocking=True))
        loss, logits = output.loss, output.logits
        acc = torch.sum(torch.argmax(logits, dim=-1) == labels.cuda(self.args.local_rank, non_blocking=True))

        return loss, float(acc) / bz, (labels.tolist(), torch.argmax(logits, dim=-1).tolist(), logits)
    

    def forward_classification_luke(self, texts, rs, labels, return_candidates=False, bz=None):
        bz = len(texts)
        input_texts = []
        input_entities = []
        input_entity_spans = []

        for i in range(bz):
            text = texts[i]
            contents = text.split('\t\t')
            e1, e2 = contents[1], contents[3]
            sentence = ' '.join(contents)
            input_texts.append(sentence)
            input_entities.append([e1, e2])
            input_entity_spans.append([(sentence.find(e1), sentence.find(e1) + len(e1)), (sentence.find(e2), sentence.find(e2) + len(e2))])
        
        encoding = self.tokenizer(
            input_texts, 
            entities=input_entities, 
            entity_spans=input_entity_spans, 
            add_prefix_space=True, 
            padding=True, 
            return_tensors="pt"
            )

        encoding = {k: v.cuda(self.args.local_rank, non_blocking=True) for k, v in encoding.items()}
        output = self.model(**encoding, labels=labels.cuda(self.args.local_rank, non_blocking=True))

        loss, logits = output.loss, output.logits
        acc = torch.sum(torch.argmax(logits, dim=-1) == labels.cuda(self.args.local_rank, non_blocking=True))

        return loss, float(acc) / bz, (labels.tolist(), torch.argmax(logits, dim=-1).tolist(), logits)


class Trainer(object):
    def prepare_gpu(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        #         print('Num of available GPUs: ', n_gpu)
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def __init__(self, args):
        self.args = args
        global device 
        device = torch.device("cuda", args.local_rank)
        self.device = device
        self.devices, self.device_ids = self.prepare_gpu(8)

        # load tokenizer
        tokenizer_src = self.args.model_name

        if self.args.model_name == 'kepler':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        elif self.args.model_name == 'luke':
            self.tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base')
        elif self.args.model_name == 'biobert':
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        elif self.args.model_name == "sapbert":
            self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        os.makedirs(self.get_save_path(), exist_ok=True)

        self.train_set, self.test_set, self.dev_set, self.link_dataset_head, self.link_dataset_tail, self.ch_test_loader, self.oh_test_loader, self.o_test_loader, relation_num = get_dataloader(
            args, self.tokenizer)
        self.model = PTuneForLAMA(args, self.device, self.device_ids, self.args.template, self.tokenizer, relation_num)

        self.model.cuda()
        if self.args.load_dir != '':
            self.load(self.args.load_dir)

    def get_task_name(self):
        str_template = [str(x) for x in self.args.template]
        str_template = '.'.join(str_template)
        names = [self.args.model_name,
                 "template_{}".format(str_template),
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, self.args.model_name, 'search', self.get_task_name())

    def get_checkpoint(self, epoch_idx, dev_f1, test_f1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_f1 * 100, 4),
                                                          round(test_f1 * 100, 4))
        return {'model_state_dict': self.model.state_dict(),
                'ckpt_name': ckpt_name,
                'dev_f1': dev_f1,
                'test_f1': test_f1}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# Checkpoint {} saved.".format(ckpt_name))

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [
            {
                'params': self.model.model.parameters(),
                 'lr': self.args.lm_lr
                 }
            ]
        # if self.args.use_lm_finetune:
        #     params.append({'params': self.model.prompt_encoder.parameters()})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        # multi-gpus
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set)
        self.dev_sampler = torch.utils.data.distributed.DistributedSampler(self.dev_set)
        # self.link_tail_sampler = torch.utils.data.distributed.DistributedSampler(self.link_dataset_tail)
        # self.link_head_sampler = torch.utils.data.distributed.DistributedSampler(self.link_dataset_head)



        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=False,
                                                        num_workers=16,
                                                        # pin_memory=True,
                                                        drop_last=True,
                                                        sampler=self.train_sampler)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.args.batch_size, sampler=self.test_sampler, drop_last=True)
        self.dev_loader = torch.utils.data.DataLoader(self.dev_set, batch_size=self.args.batch_size, sampler=self.dev_sampler, drop_last=True)
        self.link_loader_tail = torch.utils.data.DataLoader(
            self.link_dataset_tail, 
            batch_size=int(self.args.batch_size / 2), 
            # sampler=self.link_tail_sampler, 
            )
        self.link_loader_head = torch.utils.data.DataLoader(
            self.link_dataset_head,
             batch_size=int(self.args.batch_size / 2),
            #   sampler=self.link_head_sampler,
            )


        for epoch_idx in range(self.args.max_epoch):
            # run training
            self.train_sampler.set_epoch(epoch_idx)  # 这句莫忘，否则相当于没有shuffle数据
            pbar = tqdm(self.train_loader)
            self.model.train()
            for batch_idx, batch in enumerate(pbar):

                optimizer.zero_grad()
                # print(encoding)

                loss, acc, _ = self.model.forward_classification(
                    texts=batch[0],
                    # [rs.cuda(self.args.local_rank, non_blocking=True) for rs in batch[1]],
                    # labels=[label.cuda(self.args.local_rank, non_blocking=True) for label in batch[2]],
                    rs=batch[1],
                    labels=batch[2],
                    )
                pbar.set_description(f"Loss {float(loss.mean()):.6g}, acc {acc:.4g}")

                #                 print("\n begin back propagation for epoch", epoch_idx)
                # modified
                loss.backward()

                #                 print("\n end back propagation for epoch", epoch_idx)

                optimizer.step()

                loss = reduce_mean(loss, dist.get_world_size())

                # check early stopping
                if batch_idx % self.args.valid_step == 0:
                    # Triple Classification
                    dev_results, test_results = evaluate_classification_using_classification(self, epoch_idx)

                    # Link Prediction
                    if self.args.link_prediction and not (batch_idx == 0 and epoch_idx == 0):
                    # if self.args.link_prediction:
                        evaluate_link_prediction_using_classification(self, epoch_idx, batch_idx, output_scores=True)

                    # Early stop and save
                    if dev_results >= best_dev:
                        best_ckpt = self.get_checkpoint(epoch_idx, dev_results, test_results)
                        early_stop = 0
                        best_dev = dev_results
                    else:
                        early_stop += 1
                        if early_stop >= self.args.early_stop:
                            self.save(best_ckpt)
                            print("Early stopping at epoch {}.".format(epoch_idx))
                            print("FINISH_TRAIN...")
                            return best_ckpt

                    sys.stdout.flush()
            my_lr_scheduler.step()
        self.save(best_ckpt)

        return best_ckpt


def main():
    args = construct_generation_args()
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
