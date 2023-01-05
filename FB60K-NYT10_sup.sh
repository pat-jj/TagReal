export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python3 bm25.py \
--corpus_train './Volumes/Aux/Downloaded/Data-Upload/FB60K+NYT10/text/train.json' \
--corpus_test './Volumes/Aux/Downloaded/Data-Upload/FB60K+NYT10//text/test.json' \
--kg_train './dataset/FB60K-NYT10-100/train.txt' \
--kg_valid './dataset/FB60K-NYT10-100/valid.txt' \
--kg_test './dataset/FB60K-NYT10-100/test.txt' \
--t2t_out_dir './dataset/FB60K-NYT10-100/triple2text.txt' \
--q2t_out_dir_tail './dataset/FB60K-NYT10-100/query2text_tail.txt' \
--q2t_out_dir_head './dataset/FB60K-NYT10-100/query2text_head.txt' \
--sub_corpus_text_dir './dataset/FB60K-NYT10-100/sup_text.json' \
--dataset 'FB60K+NYT10' \
--entity2label './dataset/FB60K-NYT10-100/entity2label.txt' \
--tail_prediction "False" \
--head_prediction "False"
