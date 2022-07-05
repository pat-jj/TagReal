import os
import random
import json
from collections import defaultdict
from tqdm import tqdm
import pickle


NEW_DATASET = './'
NEG_RATIO = 100

# create negtive triples
triple_set = set()
domain_r = defaultdict(list)
range_r = defaultdict(list)
ent_set = set()
def handle(filenames):
    lines = open(filenames).readlines()
    for line in lines:
        h, r, t = line.strip().split('\t')
        triple_set.add((h, r, t))
        domain_r[r].append(h)
        range_r[r].append(t)
        ent_set.add(h)
        ent_set.add(t)
handle(os.path.join(NEW_DATASET, 'train.txt'))
handle(os.path.join(NEW_DATASET, 'valid.txt'))
handle(os.path.join(NEW_DATASET, 'test.txt'))
ent_list = list(ent_set)

def create_neg(filename, neg_filename, neg_ratio=1):
    lines = open(filename).readlines()
    f = open(neg_filename, 'w')
    for line in tqdm(lines):
        h, r, t = line.strip().split('\t')
        for i in range(neg_ratio):
            hh, rr, tt = h, r, t
            if random.random() < 0.5:
                find_num = 0
                while (hh, rr, tt) in triple_set and find_num < 100:
                    hh = domain_r[rr][random.randint(0, len(domain_r[rr]) - 1)]
                    find_num += 1
                if find_num >= 100:
                    while (hh, rr, tt) in triple_set:
                        hh = ent_list[random.randint(0, len(ent_list) - 1)]
            else:
                find_num = 0
                while (hh, rr, tt) in triple_set and find_num < 100:
                    tt = range_r[rr][random.randint(0, len(range_r[rr]) - 1)]
                    find_num += 1
                if find_num >= 100:
                    while (hh, rr, tt) in triple_set:
                        tt = ent_list[random.randint(0, len(ent_list) - 1)]
            f.write('{}\t{}\t{}\n'.format(hh, rr, tt))
    f.close()

create_neg(os.path.join(NEW_DATASET, 'train.txt'), os.path.join(NEW_DATASET, 'train_neg_rand.txt'), neg_ratio=NEG_RATIO)

lines = open(f'{NEW_DATASET}/train_neg_rand.txt').readlines()
random.shuffle(lines)
f = open(f'{NEW_DATASET}/train_neg_rand.txt', 'w')
for line in lines:
    f.write(line)
f.close()



# create negtive triples from KGE TuckER

lines = open('./Wiki27K.tucker.256.kge_neg.txt').readlines()
random.shuffle(lines)

f = open('./train_neg_kge_all.txt', 'w')
for line in tqdm(lines):
    if '_reverse' not in line:
        f.write(line)
    else:
        h, r, t = line.strip().split('\t')
        r = r[:r.rfind('_')]
        f.write(f'{t}\t{r}\t{h}\n')
f.close()



# for link prediction

K = 500

fh = open('link_prediction_head.txt', 'w')
ft = open('link_prediction_tail.txt', 'w')

lines = open('Wiki27K.tucker.256.test.link_prediction.txt').readlines()
for i in range(0, len(lines), K):
    h, r, t = lines[i + 0].strip().split('\t')
    if '_reverse' not in r:
        for x in range(K):
            ft.write(lines[i + x])
        ft.write('SPLIT\n')
    else:
        for x in range(K):
            h, r, t = lines[i + x].strip().split('\t')
            r = r[:-8]
            fh.write(f'{t}\t{r}\t{h}\n')
        fh.write('SPLIT\n')
ft.close()
fh.close()

