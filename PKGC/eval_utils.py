from sklearn.metrics import f1_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
import torch
from tqdm import tqdm
from collections import defaultdict
from os.path import join, abspath, dirname
from torch.utils.data import DataLoader
from data_utils.dataset import *
import numpy as np
import sys
import os

def get_f1_binary(goldens, preds, threshold):
    tmp = []
    for label in preds:
        if label < threshold:
            tmp.append(0)
        else:
            tmp.append(1)
    return f1_score(goldens, tmp, average='binary')

def evaluate_classification_without_threshold(goldens, preds):
    prec, recall, f1, _ = precision_recall_fscore_support(goldens, preds, average='binary')
    acc_num = 0
    for i in range(len(preds)):
        if preds[i] == goldens[i]:
            acc_num += 1
    acc = acc_num * 1.0 / len(preds)
    return acc, prec, recall, f1

def evaluate_classification_with_threshold(valid_goldens, valid_preds, test_goldens, test_preds, DIV_NUM=100):

    def get_final_with_thres(goldens, preds, threshold):
        tmp = []
        for label in preds:
            if label < threshold:
                tmp.append(0)
            else:
                tmp.append(1)
        prec, recall, f1, _ = precision_recall_fscore_support(goldens, tmp, average='binary')
        acc_num = 0
        for i in range(len(tmp)):
            if tmp[i] == goldens[i]:
                acc_num += 1
        acc = acc_num * 1.0 / len(tmp)
        return acc, prec, recall, f1

    max_score, min_score = max(max(valid_preds), max(test_preds)), min(min(valid_preds), min(test_preds))
    max_f1, best_thres = -100.0, 0.0

    for i in range(DIV_NUM):
        thres = min_score + (max_score - min_score) * i / float(DIV_NUM)
        this_f1 = get_f1_binary(valid_goldens, valid_preds, thres)
        if this_f1 > max_f1:
            max_f1 = this_f1
            best_thres = thres

    return get_final_with_thres(test_goldens, test_preds, best_thres)

def evaluate_classification_using_classification(self, epoch_idx):
    def evaluate(loader, evaluate_type):
        with torch.no_grad():
            labels, preds, scores, triples = [], [], [], []
            for batch in loader:
                loss, _, (labels_, preds_, logits_) = self.model.forward_classification(batch[0], batch[1], batch[2])
                labels += labels_
                preds += preds_
                logits_ = torch.nn.functional.softmax(logits_, dim=-1)
                scores = scores + logits_[:, 1].tolist()
                triples = triples + list(batch[3])
            acc, prec, recall, f1 = evaluate_classification_without_threshold(labels, preds)
            if self.args.output_cla_results:
                dataset = self.args.data_dir[self.args.data_dir.rfind('/') + 1:]
                os.makedirs('classification_results', exist_ok=True)
                with open(f'classification_results/{dataset}.{evaluate_type}.{epoch_idx}.txt', 'w') as f:
                    for i in range(len(triples)):
                        f.write('{}\t{}\t{}\n'.format(triples[i], preds[i], scores[i]))
            print("{} Epoch {}, Acc: {}, Prec: {}, Recall: {}, F1: {}".format(evaluate_type, epoch_idx, acc, prec, recall, f1))
            return f1

    self.model.eval()
    dev_f1 = evaluate(self.dev_loader, 'Dev')
    test_f1 = evaluate(self.test_loader, 'Test')

    if self.args.test_open:
        o_test_f1 = evaluate(self.o_test_loader, 'Open Test')
    return dev_f1, test_f1

def link_predicate(args, idx2score, link_triple_list, valid=False, head=False):
    lines = open(f'{args.data_dir}/entity2label.txt').readlines()
    entity2id = {lines[i].strip().split('\t')[0]:i for i in range(len(lines))}

    lines = open(f'{args.data_dir}/train.txt').readlines()
    triple_set = set()
    for line in lines:
        h, r, t = line.strip().split('\t')
        triple_set.add((h, r, t))
    
    hr2index = defaultdict(list)
    e2_idx = []
    if valid:
        lines = open(f'{args.data_dir}/valid.txt').readlines()
    else:
        lines = open(f'{args.data_dir}/test.txt').readlines()
    for index, line in enumerate(lines):
        h, r, t = line.strip().split('\t')
        if head:
            hr2index[(t, r)].append(index)
            e2_idx.append(entity2id[h])
        else:
            hr2index[(h, r)].append(index)
            e2_idx.append(entity2id[t])
    e2_idx = torch.tensor(e2_idx)

    final_score = torch.zeros(len(lines), len(entity2id)).fill_(-1e9)

    for i in range(len(link_triple_list)):
        h, r, t = link_triple_list[i].strip().split('\t')
        if (h, r, t) in triple_set:
            continue
        if head:
            for idx in hr2index[(t, r)]:
                final_score[idx][entity2id[h]] = idx2score[i]
        else:
            for idx in hr2index[(h, r)]:
                final_score[idx][entity2id[t]] = idx2score[i]
        
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])

    sort_values, sort_idxs = torch.sort(final_score, dim=1, descending=True)
    sort_idxs = sort_idxs.cpu().numpy()

    for j in range(final_score.shape[0]):
        rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
        ranks.append(rank+1)

        for hits_level in range(10):
            if rank <= hits_level:
                hits[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
    if head:
        print('Head:')
    else:
        print('Tail:')
    res = [np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0]), np.mean(ranks), np.mean(1./np.array(ranks))]
    print('Hits @10: {0}'.format(res[0]))
    print('Hits @3: {0}'.format(res[1]))
    print('Hits @1: {0}'.format(res[2]))
    print('Mean rank: {0}'.format(res[3]))
    print('Mean reciprocal rank: {0}'.format(res[4]))
    return res

def evaluate_link_prediction_using_classification(self, epoch_idx, index, output_scores=False):
    self.model.eval()
    with torch.no_grad():
        # head
        scores = []
        preds = []
        for index, batch in enumerate(tqdm(self.link_loader_head)):
            loss, _, (labels_, preds_, logits_) = self.model.forward_classification(batch[0], batch[1], batch[2])
            logits_ = torch.nn.functional.softmax(logits_, dim=-1)
            scores = scores + logits_[:, 1].tolist()
            preds = preds + preds_
        res_head = link_predicate(self.args, scores, self.link_dataset_head.triple_list, valid=False, head=True)
        if output_scores:
            model_name = 'ours'
            f = open(f'{self.args.data_dir}/{model_name}.link_prediction_head_scores.txt', 'w')
            for i in range(len(scores)):
                f.write(f'{self.link_dataset_head.triple_list[i]}\t{scores[i]}\t{preds[i]}\n')
            f.close()
        # tail
        scores = []
        preds = []
        for index, batch in enumerate(tqdm(self.link_loader_tail)):
            loss, _, (labels_, preds_, logits_) = self.model.forward_classification(batch[0], batch[1], batch[2])
            logits_ = torch.nn.functional.softmax(logits_, dim=-1)
            scores = scores + logits_[:, 1].tolist()
            preds = preds + preds_
        res_tail = link_predicate(self.args, scores, self.link_dataset_tail.triple_list, valid=False, head=False)
        if output_scores:
            model_name = 'ours'
            f = open(f'{self.args.data_dir}/{model_name}.link_prediction_tail_scores.txt', 'w')
            for i in range(len(scores)):
                f.write(f'{self.link_dataset_tail.triple_list[i]}\t{scores[i]}\t{preds[i]}\n')
            f.close()
        # avg
        res_avg = [(res_head[i] + res_tail[i]) / 2.0 for i in range(len(res_head))]
        print('Avg:')
        print('Hits @10: {0}'.format(res_avg[0]))
        print('Hits @3: {0}'.format(res_avg[1]))
        print('Hits @1: {0}'.format(res_avg[2]))
        print('Mean rank: {0}'.format(res_avg[3]))
        print('Mean reciprocal rank: {0}'.format(res_avg[4]))
