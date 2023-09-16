import os
import random
import json
from collections import defaultdict
from tqdm import tqdm
import pickle


NEW_DATASET = '.'
NEG_RATIO = 100

rel_list = ['/people/person/nationality', '/location/location/contains', '/people/person/place_lived', 
            '/people/deceased_person/place_of_death', '/people/person/ethnicity', '/people/ethnicity/people',
            '/business/person/company', '/people/person/religion', '/location/neighborhood/neighborhood_of',
            '/business/company/founders', '/people/person/children', '/location/administrative_division/country',
            '/location/country/administrative_divisions', '/business/company/place_founded', '/location/us_county/county_seat',
            '/people/person/place_of_birth']

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
create_neg(os.path.join(NEW_DATASET, 'test.txt'), os.path.join(NEW_DATASET, 'test_neg.txt'), neg_ratio=20)
create_neg(os.path.join(NEW_DATASET, 'valid.txt'), os.path.join(NEW_DATASET, 'valid_neg.txt'), neg_ratio=20)


lines = open(f'{NEW_DATASET}/train_neg_rand.txt').readlines()
random.shuffle(lines)
f = open(f'{NEW_DATASET}/train_neg_rand.txt', 'w')
for line in lines:
    f.write(line)
f.close()



# create negtive triples from KGE TuckER

lines = open('./FB60K-NYT10-sub.tucker.256.kge_neg.txt').readlines()
random.shuffle(lines)

f = open('./train_neg_kge_all.txt', 'w')
for line in tqdm(lines):
    if '_reverse' not in line:
        h, r, t = line.strip().split('\t')
        if r in rel_list:
            f.write(line)
    else:
        h, r, t = line.strip().split('\t')
        r = r[:r.rfind('_')]
        if r in rel_list:
            f.write(f'{t}\t{r}\t{h}\n')
f.close()



# for link prediction

K = 500

fh = open('link_prediction_head.txt', 'w')
ft = open('link_prediction_tail.txt', 'w')

lines = open('FB60K-NYT10-sub.tucker.256.test.link_prediction.txt').readlines()
lines_test = open('./test.txt').readlines() + open('./test.txt').readlines()
for i in range(0, len(lines), K):
    h, r, t = lines[i + 0].strip().split('\t')
    gt = lines_test[i//K]
    if '_reverse' not in r:
        ft.write(gt)
        for x in range(K):
            h, r, t = lines[i + x].strip().split('\t')
            # if r in rel_list:
            if lines[i + x] != gt:
                ft.write(lines[i + x])
        ft.write('SPLIT\n')
    else:
        fh.write(gt)
        for x in range(K):
            h, r, t = lines[i + x].strip().split('\t')
            r = r[:-8]
            # if r in rel_list:
            if f'{t}\t{r}\t{h}\n' != gt:
                fh.write(f'{t}\t{r}\t{h}\n')
        fh.write('SPLIT\n')
ft.close()
fh.close()

