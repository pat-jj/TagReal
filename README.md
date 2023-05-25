# TagReal

Source code for the paper: "[Text-Augmented Open Knowledge Graph Completion via
Pre-Trained Language Models](http://hanj.cs.illinois.edu/pdf/acl23_pjiang.pdf)" (ACL'23 Findings)


## Requirements

- python3 
- pytorch 
- transformers 

## Running KGE model and obtain recalls

TagReal adapts the "Recall and Re-ranking" framework from PKGC.

Also, we use KGE models to generate negative triples to train the PLM.

Therefore, the first step of TagReal is to run KGE models and get the recalls and negative triples.

### Dataset Downloading

We use the datasets FB60K-NYT10 and UMLS-PubMed provided by CPL (GitHub: https://github.com/INK-USC/CPL#datasets).


**Dataset generation process**:

Move to `kge_models` folder and run the KGE model (e.g., TuckER) to generate recalls and negative triples.

``` bash
cd kge_models
sh run_xxx.sh
```

Copy the following generated files into the appropriate folder.

``` bash
cp output/FB60K-NYT10.tucker.256.kge_neg.txt ../dataset/FB60K-NYT10
cp output/FB60K-NYT10.tucker.256.test.link_prediction.txt ../dataset/FB60K-NYT10
```

Move to `dataset/FB60K-NYT10` folder and run generation script to generate negative files and link prediction files.

``` bash
cd ../dataset/FB60K-NYT10
python3 pipeline.py
```

## Prompt Generation


### Prompt Selection
We use MetaPAD (https://github.com/mjiang89/MetaPAD) and TruePIE (https://github.com/qili5/TruePIE) for prompt selection.
The phrase segmentation as well as frequent pattern mining are integrated in the code of MetaPAD. 


### Sub-corpus Mining and Prompt Post-processing
Use the functions in `prompt_mining.py` for sub-corpora mining and prompt post-processing (will split the functions when releasing the code).

``` bash
cd ../..
python3 prompt_mining.py
```

### Prompt Optimization

``` bash
cd prompt_mining/prompt_opti/
python3 prompt_optim.py
```


## Support Information Retrieval

``` bash
cd ../..
sh FB60K-NYT10_sup.sh
```

Our final dataset should contain the following files.
Checklist:
``` bash
entity2label.txt                    # For FB60K-NYT10, use data_utils/FBNYT_mapping.ipynb to generate mapping. For UMLS-PubMed, use data_utils/UMLS_mapping.ipynb.
entity2label_kg.txt                 
triple2text.txt                     # Obtained by running FB60K-NYT10_sup.sh / UMLS-PubMed_sup.sh
query2text_head.txt                 # Obtained by running FB60K-NYT10_sup.sh / UMLS-PubMed_sup.sh
query2text_tail.txt                 # Obtained by running FB60K-NYT10_sup.sh / UMLS-PubMed_sup.sh
relation2template.json              # Obtained by prompt_mining.py and prompt_optim.py.
prompt_weights.json                 # Obtained by prompt_optim, only needed when run ensemble_inference.py
train.txt                           # downloaded from CPL repo
valid.txt                           # downloaded from CPL repo
test.txt                            # downloaded from CPL repo
valid_pos.txt                       # valid set (only positive triples) for triple classification, obtained by pipeline.py
valid_neg.txt                       # valid set (only negative triples) for triple classification, obtained by pipeline.py
test_pos.txt                        # test set (only positive triples) for triple classification, obtained by pipeline.py
test_neg.txt                        # test set (only negative triples) for triple classification, obtained by pipeline.py
link_prediction_head.txt            # recall and re-ranking files for head entity link prediction, obtained by pipeline.py with the recalls from KGE
link_prediction_tail.txt            # recall and re-ranking files for tail entity link prediction, obtained by pipeline.py with the recalls from KGE
train_neg_kge_all.txt               # negative triples from kge models for training, obtained by pipeline.py
train_neg_rand.txt                  # negative triples by random replacement for training, obtained by pipeline.py
```

## Run the TagReal 

Once the data is ready (all files in checklist is ready), we can run our model by following commands.

For `FB60K-NYT10`, using the following command.

``` bash
sh FB60K-NYT10.sh
```

For `UMLS-PubMed`, using the following command.

``` bash
sh UMLS-PubMed.sh
```

## Ensemble Inference

e.g., With ensemble (P_{r_1} = (P_{1_0}, P_{1_1}, ..., P_{1_n}), P_{r_2} = (P_{2_0}, P_{2_1}, ..., P_{2_n}), ..., P_{r_m} = (P_{m_0}, P_{m_1}, ..., P_{m_n})).

**Step 1**
We create ``` relation2template_{i}.json ``` with ((P_{1_i}, P_{2_i}, ... P_{m_i}) and run the TagReal n times.
Rename the link prediction (tail/head) results (output by running) to ``` ours.link_prediction_tail_scores_p{i}.txt ```

**Step 2**
Use ``` ensemble_inference.py ``` to run ensemble reference, with the obtained weight file ``` prompt_weights.json ``` which maps each prompt P_{x_y} to its weight.


### References
Xin Lv, Yankai Lin, Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu, Peng Li, Jie Zhou. Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach. *Findings of the Association for Computational Linguistics: ACL 2022.*

Cong Fu, Tong Chen, Meng Qu, Woojeong Jin, and Xiang Ren. 2019. Collaborative policy learning for open knowledge graph reasoning. *EMNLP 2019*

Li, Qi, et al. "Truepie: Discovering reliable patterns in pattern-based information extraction." *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.*

Jiang, Meng, et al. "Metapad: Meta pattern discovery from massive text corpora." *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.*
