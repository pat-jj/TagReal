export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m torch.distributed.launch --nproc_per_node=4 cli.py \
--model_name roberta-large \
--pseudo_token [PROMPT] \
--template \(1,1,1,1,1,1\) \
--max_epoch 10 \
--batch_size 16 \
--early_stop 20 \
--lr 5e-5 \
--lm_lr 1e-6 \
--seed 234 \
--decay_rate 0.99 \
--weight_decay 0.0005 \
--lstm_dropout 0.0 \
--data_dir ./dataset/nyt10 \
--out_dir ./checkpoint/nyt10 \
--valid_step 10000 \
--use_lm_finetune \
--recall_k 30 \
--pos_K 30 \
--neg_K 30 \
--random_neg_ratio 0.5 \
--keg_neg all \
--link_prediction \
--add_definition \
