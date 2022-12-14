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
    parser.add_argument("--kg_train", type=str, default='./kge_models/data/UMLS-PubMed/train.txt')
    parser.add_argument("--kg_valid", type=str, default='./Volumes/Aux/Downloaded/Data-Upload/UMLS+PubMed/kg/dev.txt')
    parser.add_argument("--kg_test", type=str, default='./Volumes/Aux/Downloaded/Data-Upload/UMLS+PubMed/kg/test.txt')
    parser.add_argument("--t2t_out_dir", type=str, default='./dataset/UMLS+PubMed/triple2text.txt')
    parser.add_argument("--q2t_out_dir_tail", type=str, default='./dataset/UMLS+PubMed/query2text_tail.txt')
    parser.add_argument("--q2t_out_dir_head", type=str, default='./dataset/UMLS+PubMed/query2text_head.txt')
    parser.add_argument("--entity_set", type=set, default=None)
    parser.add_argument("--sub_corpus_text", type=dict, default=None)
    parser.add_argument("--sub_corpus_text_dir", type=str, default='./dataset/UMLS+PubMed/sup_text.json')
    parser.add_argument("--dataset", type=str, default="UMLS+PubMed")
    parser.add_argument("--entity2label", type=str, default="./data_utils/entity2label.txt")
    parser.add_argument("--tail_prediction", type=str, default="True")
    parser.add_argument("--head_prediction", type=str, default="True")
    parser.add_argument("--train_sup", type=str, default="False")

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
                    '/location/country/administrative_divisions', '/business/company/place_founded', '/location/us_county/county_seat',
                    '/people/person/place_of_birth']
    
    return rel_list


def get_keywords(args):
    if args.dataset == "FB60K+NYT10":
        keywords = {
            '/people/person/nationality': 'nationality', '/location/location/contains': 'in', '/people/person/place_lived': 'lived', 
            '/people/deceased_person/place_of_death': 'death', '/people/person/ethnicity': 'ethnicity', '/people/ethnicity/people': 'ethnicity',
            '/business/person/company': 'works', '/people/person/religion': 'religion', '/location/neighborhood/neighborhood_of': 'neighborhood',
            '/business/company/founders': 'founder', '/people/person/children': 'children', '/location/administrative_division/country': 'administrative_division',
            '/location/country/administrative_divisions': 'administrative_division', '/business/company/place_founded': 'founded', '/location/us_county/county_seat': 'county_seat',
            '/people/person/place_of_birth': 'born'
        }
    elif args.dataset == "UMLS+PubMed":
        keywords = {
            'gene_associated_with_disease': 'associated_with',
            'disease_has_associated_gene': 'has_associated',
            'gene_mapped_to_disease': 'mapped_to',
            'disease_mapped_to_gene': 'mapped_to',
            'may_be_treated_by': 'treat',
            'may_treat': 'treat',
            'may_be_prevented_by': 'prevent',
            'may_prevent': 'prevent',
        }

    return keywords


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
            if len(sub_corpus_text[item['head']['id']]) < 500000:
                sub_corpus_text[item['head']['id']].append(tokenized_text)
            if len(sub_corpus_text[item['tail']['id']]) < 500000:
                sub_corpus_text[item['tail']['id']].append(tokenized_text)

    args.sub_corpus_text = sub_corpus_text
    
    out_file = open(args.sub_corpus_text_dir, "w")
    json.dump(sub_corpus_text, out_file)

    return sub_corpus_text

def get_entity2label(args):
    entity2label = {}
    e2l = open(args.entity2label)
    e2l = e2l.readlines()
    for item in e2l:
        if item != None:
            entity, label = item.split('\t')
            label = label[:-1]
            entity2label[entity] = label
    return entity2label

def triple2text(args):
    print("retrieve text for triples ...")
    data_corpus = get_corpus(args)
    rel_list = get_rel_list(args)
    entity_set = get_entity_set(args, data_corpus)
    entity2label = get_entity2label(args)
    keywords = get_keywords(args)
    triple_text = {}
    
    # Add the exsiting mapping
    print("load existing mapping ...")
    for item in tqdm(data_corpus):
        # try:
        if (item != None) and ('relation' in item.keys()) and (item['relation'] != None) and (item['relation'] in rel_list):
            text = item['sentence'].replace('###END###\n', '')
            head, relation, tail = item['head']['id'], item['relation'], item['tail']['id']
            triple = head + '||' + relation + '||' + tail
            if triple in triple_text.keys():
                if len(triple_text[triple]) <= 100:
                    combo = triple_text[triple] + text.lower().replace(head.lower(), head).replace(tail.lower(), tail)
                    if len(combo) <= 100:
                        triple_text[triple] = combo
            else:
                if len(text) <= 100:
                    triple_text[triple] = text.lower().replace(head.lower(), head).replace(tail.lower(), tail)
        # except:
        #     continue

    print(f"existing mapping: {len(triple_text)}")
    
    # BM25
    corpus_text = []
    print("loading corpus text ...")
    sub_corpus_text = defaultdict(list)
    print("creating sub_corpus for each entity ...")
    sub_corpus_text = get_sub_corpus_text(args, data_corpus, entity_set)

    f_train = open(args.kg_train)
    data_kg_train = f_train.readlines()

    print("BM25 searching ...")
    for line in tqdm(data_kg_train):
        # break
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
        # print(sub_text)
        if len(sub_text) == 0:
            continue

        bm25 = BM25Okapi(sub_text)
        
        text = ''
        if args.dataset == "FB60K+NYT10":
            relation = keywords[relation]
            relation = relation.replace('/', ' ').replace('_', ' ')
            query = entity2label[head] + ' ' + relation + ' ' + entity2label[tail]
        else:
            relation = relation.replace('/', ' ').replace('_', ' ')
            query = head + ' ' + relation + ' ' + tail
        tokenized_query = query.split(' ') 
        relevant_scores = bm25.get_scores(tokenized_query)
        order = np.argsort(relevant_scores)[::-1]
        for i in order:
            text = ''
            toks = sub_text[i]
            for tok in toks:
                if tok.startswith('C'):
                    tok = entity2label[tok]
                text += tok + ' '
            text = text.replace(head, entity2label[head]).replace(tail, entity2label[tail])
            if i > 0 and len(sub_text[i]) < 100 and (triple not in triple_text.keys()):
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
    triple_text = {}
    if os.path.exists(args.t2t_out_dir):
        with open(args.t2t_out_dir) as f:
            lines = f.readlines()
        for line in lines:
            line = line[:-1]
            triple, text = line.split('####SPLIT####')
            triple_text[triple] = text

    print("retrieve text for queries (in validation/testing data) ...")
    data_corpus = get_corpus(args)
    entity_set = get_entity_set(args, data_corpus) if args.entity_set == None else args.entity_set
    entity2label = get_entity2label(args)
    keywords = get_keywords(args)
    query_text_tail = {}
    query_text_head = {}

    # BM25
    corpus_text = []
    print("loading corpus text ...")
    sub_corpus_text = defaultdict(list)
    print("creating sub_corpus for each entity ...")
    sub_corpus_text = get_sub_corpus_text(args, data_corpus, entity_set)

    f_valid = open(args.kg_valid)
    f_test = open(args.kg_test)
    data_kg_valid_test = f_valid.readlines() + f_test.readlines()

    cnt_from_triple2text = 0
    for line in tqdm(data_kg_valid_test):
        items = line[:-1].split('\t')
        head, relation, tail = items[0], items[1], items[2]
        triple = head + '||' +relation + '||' + tail
        query_tail = head + '||' + relation
        query_head = tail + '||' + relation
        if triple in triple_text.keys():
            query_text_tail[query_tail] = triple_text[triple]
            query_text_head[query_head] = triple_text[triple]
            cnt_from_triple2text += 1
            continue
        if head not in sub_corpus_text.keys():
            # print(f"{head} not found in corpus")
            continue
        sub_text = sub_corpus_text[head]
        if len(sub_text) == 0:
            continue
        bm25 = BM25Okapi(sub_text)
        text = ''
        
        if args.dataset == "FB60K+NYT10":
            relation = keywords[relation]
            relation = relation.replace('/', ' ').replace('_', ' ')
            query_tail_ = entity2label[head] + ' ' + relation
            query_head_ = entity2label[tail] + ' ' + relation
        else:
            relation = relation.replace('/', ' ').replace('_', ' ')
            query_tail_ = head + ' ' + relation
            query_head_ = tail + ' ' + relation

        tokenized_query_tail = query_tail_.split(' ')
        tokenized_query_head = query_head_.split(' ')
        if args.tail_prediction == "True":
            relevant_scores_tail = bm25.get_scores(tokenized_query_tail)
        if args.head_prediction == "True":
            relevant_scores_head = bm25.get_scores(tokenized_query_head)

        if args.tail_prediction == "True":
            order = np.argsort(relevant_scores_tail)[::-1]
            for i in order:
                text = ''
                toks = sub_text[i]
                for tok in toks:
                    if tok.startswith('C'):
                        tok = entity2label[tok]
                    text += tok + ' '
                text = text.replace(head, entity2label[head]).replace(tail, entity2label[tail])
                if i > 0 and len(sub_text[i]) < 400 and query_tail not in query_text_tail.keys():
                    query_text_tail[query_tail] = text
                    break
                elif i == 0:
                    break
        if args.head_prediction == "True":
            order = np.argsort(relevant_scores_head)[::-1]
            for i in order:
                text = ''
                toks = sub_text[i]
                for tok in toks:
                    if tok.startswith('C'):
                        tok = entity2label[tok]
                    text += tok + ' '
                text = text.replace(head, entity2label[head]).replace(tail, entity2label[tail])
                if i > 0 and len(sub_text[i]) < 400 and query_head not in query_text_head.keys():
                    query_text_head[query_head] = text
                    break
                elif i == 0:
                    break
    print(f'size of query2text from triple2text: {cnt_from_triple2text}')
    out_str = ""
    for query in query_text_tail.keys():
        line = query + '####SPLIT####' + query_text_tail[query] + '\n'
        out_str += line

    out_file = open(args.q2t_out_dir_tail, 'w', encoding='utf-8')
    print(out_str, file=out_file)
    out_file.close()

    out_str = ""
    for query in query_text_head.keys():
        line = query + '####SPLIT####' + query_text_head[query] + '\n'
        out_str += line

    out_file = open(args.q2t_out_dir_head, 'w', encoding='utf-8')
    print(out_str, file=out_file)
    out_file.close()
    return 


def main():
    args = construct_args()
    print("start support information retrieval in reliable sources ...")
    if args.train_sup == "True":
        triple2text(args=args)
    query2text(args=args)


if __name__ == '__main__':
    main()