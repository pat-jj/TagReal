export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python3 -m torch.distributed.launch --nproc_per_node=7 tagreal.py \
--model_name luke \
--pseudo_token [PROMPT] \
--template \(1,1,1,1,1,1\) \
--max_epoch 10 \
--batch_size 64 \
--early_stop 10 \
--lr 5e-5 \
--lm_lr 1e-6 \
--seed 234 \
--decay_rate 0.99 \
--weight_decay 0.0005 \
--lstm_dropout 0.0 \
--data_dir ./dataset/FB60K-NYT10-100 \
--out_dir ./checkpoint/FB60K-NYT10-100 \
--valid_step 5000 \
--use_lm_finetune \
--recall_k 20 \
--pos_K 30 \
--neg_K 30 \
--random_neg_ratio 0.5 \
--kge_neg all \
--link_prediction \
--add_definition \
