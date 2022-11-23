import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def transfer_word2vec_txt_to_pickle():
    with open('../../../TruePIE/input/word_embedding.txt', 'r') as f:
        lines = f.readlines()

    df = pd.DataFrame()
    words = []
    embs = []
    for line in tqdm(lines):
        word = line.split(' ')[0]
        emb = line.split(' ')[1:]
        for i in range(len(emb)):
            emb[i] = float(emb[i])
        words.append(word)
        embs.append(emb)
    df['word'] = words
    df['emb'] = embs

    df.to_pickle('../../../TruePIE/input/word_embedding.pickle')

def truepie_data_make():

    embedding_dict = pickle.load(open('../../../TruePIE/input/word_embedding.pickle','rb'))
    pattern_file = '../../../TruePIE/input/pattern.txt'
    words = embedding_dict['word'].to_numpy()
    embs = embedding_dict['emb'].to_numpy()
    progress = 0

    def emb_of_word(word, words, embs):
        if word not in words:
            terms = word.split('_')
        else:
            idx = np.where(words == word)[0]
            return np.array(embs[idx][0])
        cap_ = ''
        for term in terms:
            cap_ += term.capitalize() + '_'
        cap_ = cap_[:-1]
        if cap_ not in words:
            return np.zeros(300)
        else:
            idx = np.where(words == cap_)[0]
            return np.array(embs[idx][0])

    emb_e_v_pairs = {}
    feature_vec1 = dict()
    ev_diff_count = defaultdict(list)
    with open(pattern_file) as f:
        patterns, pattern2patternid = [], {}
        for line in tqdm(f):
            emb_e_v_pair = {}
            row = line.rstrip('\n').split('\t')
            emb_e, emb_v = emb_of_word(row[3], words, embs), emb_of_word(row[4], words, embs)
            if (emb_e == np.zeros(300)).all() or (emb_v == np.zeros(300)).all():
                continue
            emb_e_v_pair['e'] = emb_e
            emb_e_v_pair['v'] = emb_v
            emb_e_v_pairs[row[3] + '_' + row[4]] = emb_e_v_pair

            if row[0] not in feature_vec1:
                pattern = row[0] + '\t' + row[1] + '\t' + row[2]
                row[0] = row[0].replace('_', ' ')
                pattern_t = row[0].split()
                pattern_t = [element for i, element in enumerate(pattern_t) if i not in (row[1], row[2])]

                for i in range(len(pattern_t)):
                    if pattern_t[i] == ',':
                        pattern_t[i] = '</s>'

                pattern_emb = [np.array(embs[np.where(words == w)[0]][0]) for w in pattern_t if w in words]

                if not pattern_emb:
                    continue

                if not pattern in pattern2patternid:
                    patterns.append(pattern)
                    pattern2patternid[pattern] = len(pattern2patternid)
                patternid = pattern2patternid[pattern]
                feature_vec1[patternid] =np.mean(pattern_emb, axis=0).tolist()

            patternid = pattern2patternid[pattern]
            ev_diff_count[patternid].append(((emb_e-emb_v).tolist(), row[5]))

            
    print('length of patterns: %s\n' % len(patterns))
    with open('../../../TruePIE/input/feature/'+'pattern2patternid'+ '.pickle', 'wb') as fp:
        pickle.dump(pattern2patternid, fp)
                
    with open('../../../TruePIE/input/feature/'+'patternid2pattern'+ '.pickle', 'wb') as fp:
        pickle.dump(patterns, fp)
            
    feature_vec2 = {pattern: np.mean([tpl[0] for tpl in ev_diff_count[pattern]], axis=0).tolist() for pattern in ev_diff_count}

    print('length of patterns (feature1): %s\n' % len(feature_vec1))
    with open('../../../TruePIE/input/feature/'+'feature1'+ '.pickle', 'wb') as fp:
        pickle.dump(feature_vec1, fp)
    print('length of patterns (feature2_mean): %s\n' % len(feature_vec2))
    with open('../../../TruePIE/input/feature/'+'feature2_mean'+ '.pickle', 'wb') as fp:
        pickle.dump(feature_vec2, fp)
            
        

def main():
    transfer_word2vec_txt_to_pickle()
    truepie_data_make()

    
if __name__ == "__main__":
    main()

