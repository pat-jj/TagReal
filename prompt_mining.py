from datasets import load_dataset
from tqdm import tqdm
import random
import nltk
import os

nltk.download('words')
words = set(nltk.corpus.words.words())


def get_wiki_corpus():
    wiki_dataset = load_dataset("wikipedia", "20220301.en")
    wiki_dataset = wiki_dataset['train']

    corpus_text = ""

    for context in tqdm(wiki_dataset):
        split_text = context['text'].split('.')
        for t in split_text:
            line = t + '.\n'
            corpus_text = corpus_text + line

    w_text = open("../../../data/pj20/corpus_text.txt", 'w', encoding='utf-8')
    print(corpus_text, file=w_text)

    return corpus_text


def get_pubmed_corpus():
    with open("./Volumes/pubmed_corpus_text.txt") as f:
        corpus_text = f.readlines()
    
    return corpus_text


def get_relation_set(dataset, list_=False):
    # The set of relation we're using
    valid_relation_set = set()
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

    if list_:
        return rel_list

    for i in range(len(rel_list)):
        valid_relation_set.add(rel_list[i])

    return valid_relation_set


def get_abrv_relations():
    abrv = {
        '/people/person/nationality':'ppn',
        '/location/location/contains':'llc',
        '/people/person/place_lived':'ppp',
        '/people/deceased_person/place_of_death':'pdp',
        '/people/person/ethnicity':'ppe',
        '/people/ethnicity/people':'pep',
        '/business/person/company':'bpc',
        '/people/person/religion':'ppr',
        '/location/neighborhood/neighborhood_of':'lnn',
        '/business/company/founders':'bcf',
        '/people/person/children':'ppc',
        '/location/administrative_division/country':'lac',
        '/location/country/administrative_divisions':'lca',
        '/business/company/place_founded':'bcp',
        '/location/us_county/county_seat':'luc'
    }

    return abrv


def get_triples_for_relation(relation, dataset
                            
                             # n=500
                             ):  # small: 80  big: 500
    if dataset == "FB60K-NYT10":
        original_triples_path = "./prompt_mining/triples_nyt10.txt"
    elif dataset == "UMLS-PubMed":
        original_triples_path = "./prompt_mining/triples_umls.txt"


    random_selected_triples = ""
    with open(original_triples_path) as f:
        original_triples = f.readlines()

    count = 0
    random.shuffle(original_triples)
    for idx in range(len(original_triples)):
        if relation in original_triples[idx]:
            random_selected_triples = random_selected_triples + original_triples[idx]
            count += 1

    print(f'{count} triples for {relation}')
    relation_ = relation.replace('/', '_')
    sp_path = f"./prompt_mining/relation_triples/{dataset}/" + relation_ + ".txt"
    sp_file = open(sp_path, 'w', encoding='utf-8')
    print(random_selected_triples, file=sp_file)

    return random_selected_triples


def get_entity_tokens(dataset):
    if dataset == "FB60K-NYT10":
        triples_path = "./prompt_mining/triples_nyt10.txt"
    elif dataset == "UMLS-PubMed":
        triples_path = "./prompt_mining/triples_umls.txt"

    with open(triples_path) as f:
        original_triples = f.readlines()[:-1]

    entity_set = set()
    for triple in original_triples:
        triple = triple.split('\t')
        head, tail = triple[0], triple[2][:-1]
        entity_set.add(head)
        entity_set.add(tail)

    entity_list = [*entity_set]
    entity_tokens = {}
    for i in range(len(entity_list)):
        entity_tokens[entity_list[i]] = str(i)

    return entity_tokens


def mine_triple_text_from_corpus(triples, corpus, relation, dataset, n=2000, max_lines=50000, triples_path=None):
    mined_text = ""

    with open(corpus) as f:
        corpus_lines = f.readlines()
    
    print(f"length of corpus: {len(corpus_lines)}")

    if triples_path is not None:
        with open(triples_path) as f:
            lines = f.readlines()

    else:
        lines = triples.split('\n')

    lines = lines[:-1]

    relation_ = relation.replace('/', '_')
    st_path = f"./prompt_mining/mined_text_big/{dataset}/mined_text_" + relation_ + ".txt"
    mined_text_file = open(st_path, 'w', encoding='utf-8')
    num_lines = 0
    
    for line in lines:
        # control the maximum mined text size
        if num_lines >= max_lines:
            break

        cnt = 0
        triple = line.split('\t')
        head, tail = triple[0].replace('_', ' ').strip(), triple[2].replace('_', ' ').strip()
        print('=======================')
        print(head, tail)
        print('=======================')
        for corpus_sentence in tqdm(corpus_lines):
            if (head in corpus_sentence) and (tail in corpus_sentence):
                #         and (relation in corpus_sentence) \
                mined_sentence = corpus_sentence.replace(head, '[X]').replace(tail, '[Y]') + '\n'
                #             print(mined_sentence)
                mined_text_file.write(mined_sentence)
                mined_text += mined_sentence
                cnt += 1
                num_lines += 1

                # control the maximum sentences for each triple
                if cnt == n:
                    cnt = 0
                    break

        print(f'Currently mined {num_lines} sentences')

    # relation_ = relation.replace('/', '_')
    # st_path = f"./prompt_mining/mined_text_big/{dataset}/mined_text_" + relation_ + ".txt"
    # mined_text_file = open(st_path, 'w', encoding='utf-8')
    # print(mined_text, file=mined_text_file)

    return mined_text


def label_x_and_y_with_categories(dataset, relation, head_name, tail_name, limit=False, max_length=40000):
    text_before = f'./prompt_mining/mined_text_big/{dataset}/mined_text_' + relation + ".txt"
    text_after = ""
    with open(text_before) as f:
        lines = f.readlines()

    cnt = 0
    # random.shuffle(lines)
    for line in lines:
        if line != '\n':
            # line = " ".join(w for w in nltk.wordpunct_tokenize(line) if w.lower() in words or not w.isalpha())
            line = line.lower()

            line = \
                line.replace('[x]', f'<{head_name}>[X]</{head_name}>') \
                    .replace('[y]', f'<{tail_name}>[Y]</{tail_name}>')

            text_after += line + '\n'
            cnt += 1
        if limit and cnt > max_length:
            break

    out_path = f'./prompt_mining/mined_text_big/grained/{dataset}/' + "grained_" + relation + '.txt'
    out_file = open(out_path, 'w', encoding='utf-8')
    print(text_after, file=out_file)


def filter_meta_pad_mined_results(relation, head, tail):
    meta_pad_path = './prompt_mining/metapad_mined_result/' + relation
    filter_result_path = meta_pad_path + '/filtered_patterns'

    filtered_result = ""
    lines = []
    if not os.path.exists(filter_result_path):
        os.makedirs(filter_result_path)
    if os.path.exists(meta_pad_path + '/bottom-metapattern.txt'):
        with open(meta_pad_path + '/bottom-metapattern.txt') as f:
            lines = f.readlines()
    if os.path.exists(meta_pad_path + '/top-metapattern.txt'):
        with open(meta_pad_path + '/top-metapattern.txt') as f:
            lines += f.readlines()

    pattern_set = set()
    for line in lines:
        if (head in line) and (tail in line) and ('|' not in line) and (line.split('\t')[2] not in pattern_set):
            pattern_set.add(line.split('\t')[2])
            score_pattern = line.split('\t')[1] + '\t' + line.split('\t')[2]
            filtered_result += score_pattern

    out_file = open(filter_result_path + '/result.txt', 'w', encoding='utf-8')
    print(filtered_result, file=out_file)

    return filtered_result


def from_meta_pad_to_true_pie(relation, relation_slash, patterns, entity_tokens, threshold=0.4):

    triples = get_triples_for_relation(relation_slash)

    lines = triples.split('\n')

    lines = lines[:-1]
    score_patterns = patterns.split('\n')

    output = ""
    for pattern in score_patterns:
        score = float(pattern.split('\t')[0])
        pattern = pattern.split('\t')[1]
    
        if score <= threshold:
            break

        cnt = 0
        random.shuffle(lines)
        for line in lines:
            triple = line.split('\t')
            head, tail = triple[0], triple[2]
            head_id, tail_id = entity_tokens[head], entity_tokens[tail]
            output += (pattern + '\t' + head_id + '\t' + tail_id + '\t' + head + '\t' + tail + '\t' + str(int(score*100)) + '\n')
            cnt += 1
            if cnt >= score*300:
                break

    out_file_path = './prompt_mining/truepie/input/patterns_' + relation + '.txt'
    out_file = open(out_file_path, 'w', encoding='utf-8')
    print(output, file=out_file)


def convert_type_to_x_y(relation, head, tail):
    prompt_file = './prompt_mining/truepie/output/' + relation + '_.txt'
    output_file_path = './prompt_mining/truepie/output_xy/' + relation + '.txt'
    output = ""

    with open(prompt_file) as f:
        lines = f.readlines()

    for line in lines:
        output += line.replace(f'${head}', '[X]').replace(f'${tail}', '[Y]')
    
    out_file = open(output_file_path, 'w', encoding='utf-8')
    print(output, file=out_file)
    

def main():
    # corpus = "../../../data/pj20/corpus_text_low.txt"
    dataset = "UMLS-PubMed"
    # corpus = "./Volumes/pubmed_corpus_text.txt"
    # relation_set = get_relation_set(dataset=dataset)
    # for relation in [*relation_set]:
    #     print('Begin Text Mining for Relation: ', relation)
    #     triples = get_triples_for_relation(relation, dataset)
    #     mine_triple_text_from_corpus(triples, corpus, relation, dataset)

    if dataset == "FB60K-NYT10":

        relation_entities = {
            'business_company_founders': {'head': 'COMPANY', 'tail': 'FOUNDER', 'slash': 'business/company/founders'},
            'business_company_place_founded': {'head': 'COMPANY', 'tail': 'PLACE_FOUNDED', 'slash': 'business/company/place_founded'},
            'business_person_company': {'head': 'PERSON', 'tail': 'COMPANY', 'slash': 'business/person/company'},
            'location_administrative_division_country': {'head': 'ADMINISTRATIVE_DIVISION', 'tail': 'COUNTRY', 'slash': 'location/administrative_division/country'},
            'location_country_administrative_divisions': {'head': 'COUNTRY', 'tail': 'ADMINISTRATIVE_DIVISION', 'slash': 'location/country/administrative_divisions'},
            'people_person_place_of_birth': {'head': 'PERSON', 'tail': 'LOCATION', 'slash': '/people/person/place_of_birth'},
            'location_location_contains': {'head': 'LOCATION', 'tail': 'LOCATION_SUB', 'slash': 'location/location/contains'},
            'location_neighborhood_neighborhood_of': {'head': 'LOCATION', 'tail': 'NEIGHBOR', 'slash': 'location/neighborhood/neighborhood_of'},
            'location_us_county_county_seat': {'head': 'US_COUNTY', 'tail': 'COUNTY_SEAT', 'slash': 'location/us_county/county_seat'},
            'people_deceased_person_place_of_death': {'head': 'DECEASED_PERSON', 'tail': 'PLACE_OF_DEATH', 'slash': 'people/deceased_person/place_of_death'},
            'people_ethnicity_people': {'head': 'ETHNICITY', 'tail': 'PEOPLE', 'slash': 'people/ethnicity/people'},
            'people_person_children': {'head': 'PERSON', 'tail': 'CHILDREN', 'slash': 'people/person/children'},
            'people_person_ethnicity': {'head': 'PERSON', 'tail': 'ETHNICITY', 'slash': 'people/person/ethnicity'},
            'people_person_nationality': {'head': 'PERSON', 'tail': 'NATIONALITY', 'slash': 'people/person/nationality'},
            'people_person_place_lived': {'head': 'PERSON', 'tail': 'PLACE_LIVED', 'slash': 'people/person/place_lived'},
            'people_person_religion': {'head': 'PERSON', 'tail': 'RELIGION', 'slash': 'people/person/religion'},
        }
    
    elif dataset == "UMLS-PubMed":

        relation_entities = {
            'gene_associated_with_disease': {'head': 'DISEASE', 'tail': 'GENE'},
            'disease_has_associated_gene': {'head': 'GENE', 'tail': 'DISEASE'},
            'gene_mapped_to_disease': {'head': 'DISEASE', 'tail': 'GENE'},
            'disease_mapped_to_gene': {'head': 'GENE', 'tail': 'DISEASE'},
            'may_be_treated_by': {'head': 'DRUG', 'tail': 'DISEASE'},
            'may_treat': {'head': 'DISEASE', 'tail': 'DRUG'},
            'may_be_prevented_by': {'head': 'DRUG', 'tail': 'DISEASE'},
            'may_prevent': {'head': 'DISEASE', 'tail': 'DRUG'},
        }

    for relation in relation_entities.keys():
        label_x_and_y_with_categories(
            dataset,
            relation,
            relation_entities[relation]['head'],
            relation_entities[relation]['tail']
        )

    # entity_tokens = get_entity_tokens(dataset=dataset)
    # # print(entity_tokens)
    # for relation in relation_entities.keys():
    #     filtered_patterns = filter_meta_pad_mined_results(
    #         relation,
    #         relation_entities[relation]['head'],
    #         relation_entities[relation]['tail']
    #     )
    #     relation_slash = relation_entities[relation]['slash']
    #     from_meta_pad_to_true_pie(relation, relation_slash, filtered_patterns, entity_tokens)

    # for relation in relation_entities.keys():
    #     convert_type_to_x_y(relation, relation_entities[relation]['head'], relation_entities[relation]['tail'])



if __name__ == '__main__':
    main()
