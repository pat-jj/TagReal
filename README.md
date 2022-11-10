# PKGC

Source codes and datasets for ACL 2022 Findings paper [Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach](https://aclanthology.org/2022.findings-acl.282.pdf)

## Requirements

- python3 (tested on 3.6.9)
- pytorch (tested on 1.10.0)
- transformers (tested on 4.11.3)

## Data Preparation

We put some of the data and the generation script in this github Repo. Users can download all the data directly from cloud drive or generate all the data through the generation script.

### 1. Download datasets from cloud drive

Delete the `dataset` folder. Download our dataset from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c1bd502d2b0a4a7a99eb/?dl=1) or [Google Drive](https://drive.google.com/file/d/1waE2QeVepwntuTDYNncBa6y94EPDn69C/view?usp=sharing). Put `dataset.zip` in this folder and unzip it. 

### 2. Generate datasets using script

We recommend this method for users who have difficulty downloading, for users who wish to regenerate negative examples and link prediction files, and for users who wish to use their own dataset to obtain all the files in the dataset.

**Dataset generation process** (We use the Wiki27K dataset as an example):

Move to `TuckER` folder and run the tucker model to generate files for link prediction and KGE negative files.

``` bash
cd TuckER
sh run.sh
```

Copy the following generated files into the appropriate folder.

``` bash
cp output/FB60K-NYT10.tucker.256.kge_neg.txt ../dataset/FB60K-NYT10
cp output/FB60K-NYT10.tucker.256.test.link_prediction.txt ../dataset/FB60K-NYT10
```

Move to `dataset/Wiki27K` folder and run generation script to generate negative files and link prediction files.

``` bash
cd ../dataset/Wiki27K
python3 pipeline.py
```

Our final dataset should contain the following files.
``` bash
entity2label.txt
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
o_valid_pos.txt # valid set (only positive triples) for triple classification under open-world assumption
o_valid_neg.txt # valid set (only negative triples) for triple classification under open-world assumption
o_test_pos.txt # test set (only positive triples) for triple classification under open-world assumption
o_test_neg.txt # test set (only negative triples) for triple classification under open-world assumption
link_prediction_head.txt # recall and re-ranking files for head entity link prediction
link_prediction_tail.txt # recall and re-ranking files for tail entity link prediction
train_neg_kge_all.txt # negative triples from kge models for training
train_neg_rank.txt # negative triples by random replacement for training
```

## Run our model

For `FB15K-237-N`, using the following command.

``` bash
sh 237.sh
```

For `Wiki27K`, using the following command.

``` bash
sh wiki27k.sh
```

## Cite

If you use the code, please cite this paper:

Xin Lv, Yankai Lin, Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu, Peng Li, Jie Zhou. Do Pre-trained Models Benefit Knowledge Graph Completion? A Reliable Evaluation and a Reasonable Approach. *Findings of the Association for Computational Linguistics: ACL 2022.*
