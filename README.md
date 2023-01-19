# TagReal

TagReal reuses some modules from PKGC (GitHub: https://github.com/THU-KEG/PKGC). So, there will be some similar steps between those two frameworks.


## Requirements

- python3 
- pytorch 
- transformers 

## Running KGE model and obtain recalls

TagReal adapts the "Recall and Re-ranking" framework from PKGC.

Also, we use KGE models to generate negative triples to train the PLM.

Therefore, the first step of TagReal is to run KGE models and get the recalls and negative triples.

### Dataset Downloading

We use the datasets FB60K-NYT10 and UMLS-PubMed provided by CPL (Cong Fu, Tong Chen, Meng Qu, Woojeong Jin, and Xiang Ren. 2019. Collaborative policy learning for open knowledge graph reasoning.
The datasets can be downloaded from their GitHub repo: https://github.com/INK-USC/CPL#datasets).


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


### Support Information Retrieval

``` bash
cd ../..
python3 bm25.py
```


Our final dataset should contain the following files.
``` bash
entity2label.txt
entity2label_kg.txt
triple2text.txt
query2text_head.txt
query2text_tail.txt
entity2definition.txt
relation2label.json
relation2template.json
train.txt
valid.txt
test.txt
valid_pos.txt # valid set (only positive triples) for triple classification under closed-world assumption
valid_neg.txt # valid set (only negative triples) for triple classification under closed-world assumption
test_pos.txt # test set (only positive triples) for triple classification under closed-world assumption
test_neg.txt # test set (only negative triples) for triple classification under closed-world assumption
link_prediction_head.txt # recall and re-ranking files for head entity link prediction
link_prediction_tail.txt # recall and re-ranking files for tail entity link prediction
train_neg_kge_all.txt # negative triples from kge models for training
train_neg_rand.txt # negative triples by random replacement for training
```

### Run our model

For `FB60K-NYT10`, using the following command.

``` bash
sh FB60K-NYT10.sh
```

For `Wiki27K`, using the following command.

``` bash
sh UMLS-PubMed.sh
```

### References
Xin Lv, Yankai Lin, Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu, Peng Li, Jie Zhou. Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach. *Findings of the Association for Computational Linguistics: ACL 2022.*

Cong Fu, Tong Chen, Meng Qu, Woojeong Jin, and Xiang Ren. 2019. Collaborative policy learning for open knowledge graph reasoning. *EMNLP 2019*

Li, Qi, et al. "Truepie: Discovering reliable patterns in pattern-based information extraction." *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.*

Jiang, Meng, et al. "Metapad: Meta pattern discovery from massive text corpora." *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.*
