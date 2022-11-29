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


def get_relation_set():
    # The set of relation we're using
    valid_relation_set = set()

    rel_list = [
        '/people/person/nationality',
        '/location/location/contains',
        '/people/person/place_lived',
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

    for i in range(len(rel_list)):
        valid_relation_set.add(rel_list[i])

    return valid_relation_set


def get_triples_for_relation(relation,
                             # n=500
                             ):  # small: 80  big: 500
    original_triples_path = "./prompt_mining/triples_nyt10.txt"

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
    sp_path = "./prompt_mining/relation_triples/triples_nyt10" + relation_ + ".txt"
    sp_file = open(sp_path, 'w', encoding='utf-8')
    print(random_selected_triples, file=sp_file)

    return random_selected_triples


def get_entity_tokens():
    triples_path = "./prompt_mining/triples_nyt10.txt"

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


def mine_triple_text_from_corpus(triples, corpus, relation, n=2000, max_lines=100000, triples_path=None):
    mined_text = ""

    with open(corpus) as f:
        corpus_lines = f.readlines()

    if triples_path is not None:
        with open(triples_path) as f:
            lines = f.readlines()

    else:
        lines = triples.split('\n')

    lines = lines[:-1]

    num_lines = 0
    for line in lines:
        # control the maximum mined text size
        if num_lines >= max_lines:
            break

        cnt = 0
        triple = line.split('\t')
        head, tail = triple[0].replace('_', ' '), triple[2].replace('_', ' ')
        print('=======================')
        print(head, tail)
        print('=======================')
        for corpus_sentence in corpus_lines:
            if (head in corpus_sentence) and (tail in corpus_sentence):
                #         and (relation in corpus_sentence) \
                mined_sentence = corpus_sentence.replace(head, '[X]').replace(tail, '[Y]')
                #             print(mined_sentence)

                mined_text = mined_text + mined_sentence + '\n'
                cnt += 1
                num_lines += 1

                # control the maximum sentences for each triple
                if cnt == n:
                    cnt = 0
                    break

        print(f'Currently mined {num_lines} sentences')

    relation_ = relation.replace('/', '_')
    st_path = "./prompt_mining/mined_text_big/mined_text" + relation_ + ".txt"
    mined_text_file = open(st_path, 'w', encoding='utf-8')
    print(mined_text, file=mined_text_file)

    return mined_text


def label_x_and_y_with_categories(relation, head_name, tail_name, limit=False, max_length=40000):
    text_before = './prompt_mining/mined_text_big/mined_text_' + relation + ".txt"
    text_after = ""
    with open(text_before) as f:
        lines = f.readlines()

    cnt = 0
    # random.shuffle(lines)
    for line in lines:
        if line != '\n':
            line = " ".join(w for w in nltk.wordpunct_tokenize(line) if w.lower() in words or not w.isalpha())

            line = \
                line.replace('[ X ]', f'<{head_name}>[X]</{head_name}>') \
                    .replace('[ Y ]', f'<{tail_name}>[Y]</{tail_name}>')

            text_after += line + '\n'
            cnt += 1
        if limit and cnt > max_length:
            break

    out_path = './prompt_mining/mined_text_big/grained/' + "grained_" + relation + '.txt'
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
    relation_set = get_relation_set()
    # for relation in [*relation_set]:
    #     print('Begin Text Mining for Relation: ', relation)
    #     triples = get_triples_for_relation(relation)
    #     mine_triple_text_from_corpus(triples, corpus, relation)

    relation_entities = {
        'business_company_founders': {'head': 'COMPANY', 'tail': 'FOUNDER', 'slash': 'business/company/founders'},
        'business_company_place_founded': {'head': 'COMPANY', 'tail': 'PLACE_FOUNDED', 'slash': 'business/company/place_founded'},
        'business_person_company': {'head': 'PERSON', 'tail': 'COMPANY', 'slash': 'business/person/company'},
        'location_administrative_division_country': {'head': 'ADMINISTRATIVE_DIVISION', 'tail': 'COUNTRY', 'slash': 'location/administrative_division/country'},
        'location_country_administrative_divisions': {'head': 'COUNTRY', 'tail': 'ADMINISTRATIVE_DIVISION', 'slash': 'location/country/administrative_divisions'},
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

    for relation in relation_entities.keys():
        label_x_and_y_with_categories(
            relation,
            relation_entities[relation]['head'],
            relation_entities[relation]['tail']
        )

    entity_tokens = get_entity_tokens()
    # print(entity_tokens)
    for relation in relation_entities.keys():
        filtered_patterns = filter_meta_pad_mined_results(
            relation,
            relation_entities[relation]['head'],
            relation_entities[relation]['tail']
        )
        relation_slash = relation_entities[relation]['slash']
        from_meta_pad_to_true_pie(relation, relation_slash, filtered_patterns, entity_tokens)

    for relation in relation_entities.keys():
        convert_type_to_x_y(relation, relation_entities[relation]['head'], relation_entities[relation]['tail'])



if __name__ == '__main__':
    main()
