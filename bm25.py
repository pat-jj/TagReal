from rank_bm25 import BM25Okapi
import argparse
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import os

def construct_args():

    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--corpus_train", type=str, default='./Volumes/Aux/Downloaded/Data-Upload/UMLS+PubMed/text/train.json')
    parser.add_argument("--corpus_test", type=str, default='./Volumes/Aux/Downloaded/Data-Upload/UMLS+PubMed/text/test_5sent.json')
    parser.add_argument("--kg_train", type=str, default='./TuckER/data/UMLS-PubMed/train.txt')
    parser.add_argument("--kg_valid", type=str, default='./Volumes/Aux/Downloaded/Data-Upload/UMLS+PubMed/kg/dev.txt')
    parser.add_argument("--kg_test", type=str, default='./Volumes/Aux/Downloaded/Data-Upload/UMLS+PubMed/kg/test.txt')
    parser.add_argument("--t2t_out_dir", type=str, default='./dataset/UMLS+PubMed/triple2text.txt')
    parser.add_argument("--q2t_out_dir", type=str, default='./dataset/UMLS+PubMed/query2text.txt')
    parser.add_argument("--entity_set", type=set, default=None)
    parser.add_argument("--sub_corpus_text", type=dict, default=None)
    parser.add_argument("--sub_corpus_text_dir", type=str, default='./dataset/UMLS+PubMed/sup_text.json')
    parser.add_argument("--dataset", type=str, default="UMLS+PubMed")

    args = parser.parse_args()

    return args

def get_entity_set(args, data_corpus):
    print("constructing entity set ...")
    entity_set = set()
    f_train = open(args.kg_train)
    f_test = open(args.kg_test)
    f_valid = open(args.kg_valid)
    data_kg_raw = f_train.readlines() + f_test.readlines() + f_valid.readlines()
    dc = data_corpus

    for item in tqdm(dc):
        head, tail, relation = item['head']['id'], item['tail']['id'], item['relation']
        entity_set.add(head)
        entity_set.add(tail)

    for line in tqdm(data_kg_raw):
        items = line[:-1].split('\t')
        head, relation, tail = items[0], items[1], items[2]
        entity_set.add(head)
        entity_set.add(tail)
    
    args.entity_set = entity_set

    return entity_set



def get_corpus(args):
    print("loading corpus ...")
    corpus_train = json.load(open(args.corpus_train))
    corpus_test = json.load(open(args.corpus_test))
    return corpus_train + corpus_test


def get_rel_list(args):
    print("getting valid relation list ...")
    rel_list = None
    if args.dataset == "UMLS+PubMed":
        rel_list = [
            'gene_associated_with_disease',
            'disease_has_associated_gene',
            'gene_mapped_to_disease',
            'disease_mapped_to_gene',
            'may_be_treated_by',
            'may_treat',
            'may_be_prevented_by',
            'may_prevent',
        ]
    elif args.dataset == "FB60K+NYT10":
        rel_list = ['/people/person/nationality', '/location/location/contains', '/people/person/place_lived', 
            '/people/deceased_person/place_of_death', '/people/person/ethnicity', '/people/ethnicity/people',
            '/business/person/company', '/people/person/religion', '/location/neighborhood/neighborhood_of',
            '/business/company/founders', '/people/person/children', '/location/administrative_division/country',
            '/location/country/administrative_divisions', '/business/company/place_founded', '/location/us_county/county_seat']
    
    return rel_list


def corpus2lower(tokenized_corpus_text, entity_set):    
    tokens = tokenized_corpus_text
    for j in range(len(tokens)):
        if tokens[j] in entity_set:
            continue
        else:
            tokens[j] = tokens[j].lower()
    return tokenized_corpus_text


def get_sub_corpus_text(args, data_corpus, entity_set):
    if os.path.exists(args.sub_corpus_text_dir):
        print("found existing constructed support corpus, loading ...")
        return json.load(open(args.sub_corpus_text_dir))
    
    sub_corpus_text = defaultdict(list)

    for item in tqdm(data_corpus):
        if item['head']['id'] in entity_set and item['tail']['id'] in entity_set:
            tokenized_text = item['sentence'].replace('###END###\n', '').split(" ")
            tokenized_text = corpus2lower(tokenized_text, entity_set)
            if len(sub_corpus_text[item['head']['id']]) < 100000:
                sub_corpus_text[item['head']['id']].append(tokenized_text)
            if len(sub_corpus_text[item['tail']['id']]) < 100000:
                sub_corpus_text[item['tail']['id']].append(tokenized_text)

    args.sub_corpus_text = sub_corpus_text
    
    out_file = open(args.sub_corpus_text_dir, "w")
    json.dump(sub_corpus_text, out_file)

    return sub_corpus_text


def triple2text(args):
    print("retrieve text for triples (in training data) ...")
    data_corpus = get_corpus(args)
    rel_list = get_rel_list(args)
    entity_set = get_entity_set(args, data_corpus)
    triple_text = {}
    
    # Add the exsiting mapping
    for item in data_corpus:
        try:
            if (item != None) and ('relation' in item.keys()) and (item['relation'] != None) and (item['relation'] in rel_list):
                text = item['sentence'].replace('###END###\n', '')
                head, relation, tail = item['head']['id'], item['relation'], item['tail']['id']
                triple = head + '||' + relation + '||' + tail
                if triple in triple_text.keys():
                    if len(triple_text[triple]) <= 200:
                        triple_text[triple] = triple_text[triple] + text.lower().replace(head.lower(), head).replace(tail.lower(), tail)
                else:
                    if len(text) <= 200:
                        triple_text[triple] = text.lower().replace(head.lower(), head).replace(tail.lower(), tail)
        except:
            continue
    
    # BM25
    corpus_text = []
    print("loading corpus text ...")
    sub_corpus_text = defaultdict(list)
    print("creating sub_corpus for each entity ...")
    sub_corpus_text = get_sub_corpus_text(args, data_corpus, entity_set)

    f_train = open(args.kg_train)
    data_kg_train = f_train.readlines()
    triple_text = {}

    print("BM25 searching ...")
    for line in tqdm(data_kg_train):
        items = line[:-1].split('\t')
        head, relation, tail = items[0], items[1], items[2]
        triple = head + '||' + relation + '||' + tail
        if head not in sub_corpus_text.keys() and tail not in sub_corpus_text.keys():
            continue
        if head not in sub_corpus_text.keys():
            # print(f"{head} not found in corpus")
            sub_corpus_text[head] = []
        if tail not in sub_corpus_text.keys():
            # print(f"{tail} not found in corpus")
            sub_corpus_text[tail] = []
        sub_text = sub_corpus_text[head]
        sub_text.extend(sub_corpus_text[tail])
        # print(sub_text)
        if len(sub_text) == 0:
            continue

        bm25 = BM25Okapi(sub_text)
        
        text = ''
        relation = relation.replace('/', ' ').replace('_', ' ')
        query = head + ' ' + relation + ' ' + tail
        tokenized_query = query.split(' ') 
        relevant_scores = bm25.get_scores(tokenized_query)
        order = np.argsort(relevant_scores)[::-1]
        for i in order:
            text = sub_text[i]
            if i > 0 and len(sub_text[i]) < 400:
                triple_text[triple] = text
                break
            elif i == 0:
                break
    
    out_str = ""
    for triple in triple_text.keys():
        line = triple + '####SPLIT####' + triple_text[triple] + '\n'
        out_str += line
    out_file = open(args.t2t_out_dir, 'w', encoding='utf-8')
    print(out_str, file=out_file)
    return 


def query2text(args):
    print("retrieve text for queries (in validation/testing data) ...")
    data_corpus = get_corpus(args)
    entity_set = get_entity_set(args) if args.entity_set == None else args.entity_set
    query_text = {}

    # BM25
    corpus_text = []
    print("loading corpus text ...")
    sub_corpus_text = defaultdict(list)
    print("creating sub_corpus for each entity ...")
    sub_corpus_text = get_sub_corpus_text(args, data_corpus, entity_set)

    f_valid = open(args.kg_valid)
    f_test = open(args.kg_test)
    data_kg_valid_test = f_valid.readlines() + f_test.readlines()

    for line in tqdm(data_kg_valid_test):
        items = line[:-1].split('\t')
        head, relation, tail = items[0], items[1], items[2]
        query = head + '||' + relation
        if head not in sub_corpus_text.keys():
            # print(f"{head} not found in corpus")
            continue
        sub_text = sub_corpus_text[head]
        if len(sub_text) == 0:
            continue
        bm25 = BM25Okapi(sub_text)
        text = ''
        relation = relation.replace('/', ' ').replace('_', ' ')
        query = head + ' ' + relation
        tokenized_query = query.split(' ')
        relevant_scores = bm25.get_scores(tokenized_query)
        order = np.argsort(relevant_scores)[::-1]
        for i in order:
            text = sub_text[i]
            if i > 0 and len(sub_text[i]) < 400:
                query_text[query] = text
                break
            elif i == 0:
                break
    
    out_str = ""
    for query in query_text.keys():
        line = query + '####SPLIT####' + query_text[query] + '\n'
        out_str += line

    out_file = open(args.q2t_out_dir, 'w', encoding='utf-8')
    print(out_str, file=out_file)
    return 


def main():
    args = construct_args()
    print("start support information retrieval in reliable sources ...")
    triple2text(args=args)
    query2text(args=args)


if __name__ == '__main__':
    main()