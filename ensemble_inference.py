import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

def read_lp_result(file_name):
    head_rel = []
    tail_rel = []
    head = []
    tail = []
    rel = []
    score = []
    pred = []

    with open(file_name) as f:
        lines = f.readlines()
    for line in lines:
        h, r, t, s, p = line.split('\t')
        p = p[:-1]
        hr = (h, r)
        tr = (t, r)

        head_rel.append(hr)
        tail_rel.append(tr)
        head.append(h)
        tail.append(t)
        rel.append(r)
        score.append(float(s))
        pred.append(int(p))
    
    d = {'hr':head_rel, 't': tail, 'tr': tail_rel, 'h': head, 'r': rel, 'score':score, 'pred':pred}
    return pd.DataFrame.from_dict(d)


def get_relation_set(dataset):
    rel_list = []
    if dataset == "FB60K-NYT10":
        rel_list = [
            '/people/person/nationality',
            '/location/location/contains',
            '/people/person/place_lived',
            '/people/person/place_of_birth',
            '/people/deceased_person/place_of_death',
            '/people/person/ethnicity',
            '/people/ethnicity/people',
            '/business/person/company',
            '/people/person/religion',
            '/location/neighborhood/neighborhood_of',
            '/business/company/founders',
            '/people/person/children',
            '/location/administrative_division/country',
            '/location/country/administrative_divisions',
            '/business/company/place_founded',
            '/location/us_county/county_seat'
        ]
    if dataset == "UMLS-PubMed":
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

    return rel_list

def load_prompt_weights(dataset):
    weights_file = f'./dataset/{dataset}/prompt_weights.json'
    with open(weights_file) as f:
        weights = json.load(f)
    return weights


def inference(df, head=False, rel='all', prompts=None, weights=None, data_dir="", scores=None):
    test_lines = open(data_dir + '/test.txt').readlines()
    if rel != 'all':
        # print(rel)
        df = df[df['r']==rel]
        new_lines = []
        for line in test_lines:
            if rel in line:
                new_lines.append(line)
        test_lines = new_lines

    if head:
        for line in test_lines:
            h, r, t = line.split('\t')
            t = t[:-1]
            query = t + '||' + r
            if query not in scores.keys():
                scores[query] = defaultdict()
            df_tmp = df[df['tr']==(t, r)].sort_values('score', ascending=False).reset_index()
            for index, row in df_tmp.iterrows():
                if row['h'] in scores[query].keys():
                    scores[query][row['h']] += row['score'] * weights[prompts[row['r']]]
                else:
                    scores[query][row['h']] = row['score'] * weights[prompts[row['r']]]
    
    else:
        for line in test_lines:
            h, r, t = line.split('\t')
            t = t[:-1]
            query = h + '||' + r
            if query not in scores.keys():
                scores[query] = defaultdict()            
            df_tmp = df[df['hr']==(h, r)].sort_values('score', ascending=False).reset_index()
            for index, row in df_tmp.iterrows():
                if row['t'] in scores[query].keys():
                    scores[query][row['t']] += row['score'] * weights[prompts[row['r']]]
                else:
                    scores[query][row['t']] = row['score'] * weights[prompts[row['r']]]

    return scores


def main():
    dataset = 'FB60K-NYT10'
    # dataset = 'UMLS-PubMed'
    prompts_num_for_rel = 5
    path = f'./dataset/{dataset}/'
    weights = load_prompt_weights(dataset=dataset)
    rel_list = get_relation_set(dataset=dataset)

    prompts_template = {}
    prompts_tail = {}
    prompts_head = {}
    for i in range(len(prompts_num_for_rel)):
        prompts_template[i] = path + f"relation2template_{i}.json "
        prompts_tail[i] = path + f"ours.link_prediction_tail_scores_p{i}.txt"
        prompts_head[i] = path + f"ours.link_prediction_head_scores_p{i}.txt"

    
    # tail 
    scores_tail = defaultdict(dict)
    for i in range(len(prompts_num_for_rel)):
        df = read_lp_result(prompts_tail[i])
        scores_tail = inference(df=df, head=False, prompts=prompts_template[i], weights=weights, data_dir=path, scores=scores_tail)

    # head 
    scores_head = defaultdict(dict)
    for i in range(len(prompts_num_for_rel)):
        df = read_lp_result(prompts_head[i])
        scores_head = inference(df=df, head=True, prompts=prompts_template[i], weights=weights, data_dir=path, scores=scores_head)

    
    # sort the results
    test_lines = open(path + '/test.txt').readlines()

    hits_tail = []
    hits_head = []
    ranks_tail = []
    ranks_head = []
    for i in range(10):
        hits_tail.append([])
        hits_head.append([])


    for line in test_lines:
        h, r, t = line.split('\t')
        t = t[:-1]
        tail_query = h + '||' + r
        head_query = t + '||' + r
        tail_truth = t
        head_truth = h

        sorted_score_tail = sorted(scores_tail[tail_query], key=scores_tail[tail_query].get)
        sorted_score_head = sorted(scores_head[head_query], key=scores_head[head_query].get)

        rank_tail = sorted_score_tail.index(tail_truth)
        rank_head = sorted_score_head.index(head_truth)

        ranks_tail.append(rank_tail+1)
        ranks_head.append(rank_head+1)

        for hits_level in range(10):

            if rank_tail <= hits_level:
                hits_tail[hits_level].append(1.0)
            else:
                hits_tail[hits_level].append(0.0)

            if rank_head <= hits_level:
                hits_head[hits_level].append(1.0)
            else:
                hits_head[hits_level].append(0.0)

    print("Tail Prediction:")
    print('Hits @10: {0}'.format(np.mean(hits_tail[9])))
    print('Hits @5: {0}'.format(np.mean(hits_tail[4])))
    print('Hits @1: {0}'.format(np.mean(hits_tail[0])))
    print('Mean rank: {0}'.format(np.mean(ranks_tail)))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_tail))))

    print("Head Prediction:")
    print('Hits @10: {0}'.format(np.mean(hits_head[9])))
    print('Hits @5: {0}'.format(np.mean(hits_head[4])))
    print('Hits @1: {0}'.format(np.mean(hits_head[0])))
    print('Mean rank: {0}'.format(np.mean(ranks_head)))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks_head))))


if __name__ == '__main__':
    main()