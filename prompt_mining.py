from datasets import load_dataset
from tqdm import tqdm
import random


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
        # '/location/location/contains',
        '/people/person/place_lived',
        # '/people/deceased_person/place_of_death',
        # '/people/person/ethnicity',
        # '/people/ethnicity/people',
        # '/business/person/company',
        '/people/person/religion',
        # '/location/neighborhood/neighborhood_of',
        # '/business/company/founders',
        # '/people/person/children',
        # '/location/administrative_division/country',
        # '/location/country/administrative_divisions',
        # '/business/company/place_founded',
        # '/location/us_county/county_seat'
    ]

    for i in range(len(rel_list)):
        valid_relation_set.add(rel_list[i])

    return valid_relation_set


def get_triples_for_relation(relation,
                             # n=500
                             ):                  # small: 80  big: 500
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


def mine_triple_text_from_corpus(triples, corpus, relation, n=500, max_lines=100000, triples_path=None):
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

    relation_ = relation.replace('/', '_')
    st_path = "./prompt_mining/mined_text_big/mined_text" + relation_ + ".txt"
    mined_text_file = open(st_path, 'w', encoding='utf-8')
    print(mined_text, file=mined_text_file)

    return mined_text


def label_x_and_y_with_categories(text_before, head_name, tail_name):

    text_after = ""
    with open(text_before) as f:
        lines = f.readlines()

    for line in lines:
        line = \
            line.replace('[X]', f'<{head_name}>[X]</{head_name}>')\
                .replace('[Y]', f'<{tail_name}>[Y]</{tail_name}>')
        text_after += line

    out_path = text_before[:-4] + '_.txt'
    out_file = open(out_path, 'w', encoding='utf-8')
    print(text_after, file=out_file)


def main():

    corpus = "../../../data/pj20/corpus_text_low.txt"
    relation_set = get_relation_set()
    for relation in [*relation_set]:
        print('Begin Text Mining for Relation: ', relation)
        triples = get_triples_for_relation(relation)
        mine_triple_text_from_corpus(triples, corpus, relation)

    # label_x_and_y_with_categories('./prompt_mining/mined_text_big/mined_text_location_neighborhood_neighborhood_of.txt',
    #                               'LOCATION', 'NEIGHBOR')


if __name__ == '__main__':
    main()
