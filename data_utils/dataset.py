from torch.utils.data import Dataset
import json
import os
import random
import pickle
from collections import defaultdict
from data_utils.vocab import get_vocab_by_strategy, token_wrapper

class BasicDataWiki:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.dataset = args.data_dir
        self.tokenizer = tokenizer
        self.init_templates()
        self.init_definition()

    def init_definition(self):
        if os.path.exists(os.path.join(self.dataset, 'entity2definition.txt')) and self.args.add_definition:
            self.entity2definition = {}
            lines = open(os.path.join(self.dataset, 'entity2definition.txt'))
            for line in lines:
                h, d = line.strip().split('\t')
                self.entity2definition[h] = d
        else:
            self.entity2definition = None

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
        self.entity2definition = basic_data.entity2definition

        self.triples = None

    def convert_from_triple_to_sentence(self, triple):
        h, r, t = triple

        this_template = self.relation2template[r].strip()

        if self.entity2definition is not None:
            def_h = self.entity2definition[h] if h in self.entity2definition else 'None'
            def_t = self.entity2definition[t] if t in self.entity2definition else 'None'
            this_template = f'The definition of {self.entity2label[h]} : {def_h} . The definition of {self.entity2label[t]} : {def_t} . {this_template}'

        if self.entity2label is not None:
            h, t = self.entity2label[h], self.entity2label[t]

        this_template = this_template.replace('[X]', '::;;##').replace('[Y]', '::;;##')
        prompts = this_template.split('::;;##')
        prompts = [x.strip() for x in prompts]
        assert(len(prompts) == 3)

        idx_x = self.relation2template[r].find('[X]')
        idx_y = self.relation2template[r].find('[Y]')
        if idx_x < idx_y:
            final_list = [prompts[0], h.strip(), prompts[1], t.strip(), prompts[2]]
        else:
            final_list = [prompts[0], t.strip(), prompts[1], h.strip(), prompts[2]]
        return '\t\t'.join(final_list)

    def __getitem__(self, i):
        if self.triples is None:
            return self.texts[i], self.rs[i], self.labels[i]
        else:
            return self.texts[i], self.rs[i], self.labels[i], self.triples[i]

    def __len__(self):
        return len(self.labels)

class KEDatasetWiki(BasicDatasetWiki):
    def __init__(self, pos_file, neg_file_random, basic_data, neg_file_kge=None, pos_K=1, neg_K=1, random_neg_ratio=1.0):
        super().__init__(basic_data)
        self.pos_K = pos_K
        self.neg_K = neg_K
        self.random_neg_ratio = random_neg_ratio
        self.texts, self.rs, self.labels, self.triples = self.process_data(pos_file, neg_file_random, neg_file_kge)

    def process_data(self, pos_file, neg_file_random, neg_file_kge):
        relation_list = []
        texts, rs, labels, triples = [], [], [], []
        pos_lines = open(pos_file).readlines()
        neg_rand_lines = open(neg_file_random).readlines()
        if neg_file_kge is not None:
            neg_kge_lines = open(neg_file_kge).readlines()
        #     random.shuffle(neg_kge_lines)
        # WARNING: data must be shuffled
        # random.shuffle(neg_rand_lines)
        rand_neg_k = int(self.neg_K * self.random_neg_ratio)
        kge_neg_k = self.neg_K - rand_neg_k
        for i in range(len(pos_lines)):
            pos_triple = pos_lines[i].strip().split('\t')
            for x in range(self.pos_K):
                texts.append(self.convert_from_triple_to_sentence(pos_triple))
                labels.append(1)
                rs.append(self.relation2idx[pos_triple[1]])
                triples.append('\t'.join(pos_triple))
            for x in range(rand_neg_k * i, rand_neg_k * (i + 1)):
                neg_triple = neg_rand_lines[x].strip().split('\t')
                texts.append(self.convert_from_triple_to_sentence(neg_triple))
                labels.append(0)
                rs.append(self.relation2idx[neg_triple[1]])
                triples.append('\t'.join(neg_triple))
            for x in range(kge_neg_k * i, kge_neg_k * (i + 1)):
                neg_triple = neg_kge_lines[x].strip().split('\t')
                texts.append(self.convert_from_triple_to_sentence(neg_triple))
                labels.append(0)
                rs.append(self.relation2idx[neg_triple[1]])
                triples.append('\t'.join(neg_triple))
        return texts, rs, labels, triples

class KEDatasetWikiInfer(BasicDatasetWiki):
    def __init__(self, filename, basic_data, recall_k):
        super().__init__(basic_data)
        self.get_lines(filename, recall_k)
        self.texts, self.rs, self.labels = self.process_data(filename)

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

    def process_data(self, filename):
        texts, rs, labels = [], [], []
        for i in range(len(self.triple_list)):
            pos_triple = self.triple_list[i].strip().split('\t')
            texts.append(self.convert_from_triple_to_sentence(pos_triple))
            labels.append(1)
            rs.append(self.relation2idx[pos_triple[1]])
        return texts, rs, labels