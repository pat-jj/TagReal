import json
import os
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
from os.path import join, abspath, dirname
from eval_utils import *
from data_utils.dataset import *
from data_utils.utils import *
from p_tuning.modeling import PTuneForLAMA
from transformers import LukeTokenizer

SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased', 'bert-base-uncased', 'bert-large-uncased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large', 'luke', 'kepler',
                  'megatron_11b']


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--model_name", type=str, default='roberta-large', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--template", type=str, default="(5, 5, 5)")
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--valid_step", type=int, default=10000)
    parser.add_argument("--recall_k", type=int, default=64)
    parser.add_argument("--pos_K", type=int, default=1)
    parser.add_argument("--neg_K", type=int, default=1)
    parser.add_argument("--random_neg_ratio", type=float, default=1.0)
    parser.add_argument("--keg_neg", type=str, default='all', choices=['all', 'tail'])

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lm_lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # lama configuration
    parser.add_argument("--use_original_template", action='store_true')
    parser.add_argument("--use_lm_finetune", action='store_true')

    parser.add_argument("--link_prediction", action='store_true')
    parser.add_argument("--output_cla_results", action='store_true')
    parser.add_argument("--add_definition", action='store_true')
    parser.add_argument("--test_open", action='store_true')

    # directories
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--out_dir", type=str, default='./dataset')
    parser.add_argument("--load_dir", type=str, default='')

    args = parser.parse_args()

    # post-parsing args

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda:0'

        # load tokenizer
        tokenizer_src = self.args.model_name

        if self.args.model_name == 'kepler':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        elif self.args.model_name == 'luke':
            self.tokenizer = LukeTokenizer.from_pretrained('studio-ousia/luke-base')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        os.makedirs(self.get_save_path(), exist_ok=True)
        self.train_loader, self.dev_loader, self.test_loader, self.ch_test_loader, self.oh_test_loader, self.o_test_loader, self.link_loader_head, self.link_loader_tail, relation_num, self.link_dataset_head, self.link_dataset_tail = get_dataloader(args, self.tokenizer)
        self.model = PTuneForLAMA(args, self.device, self.args.template, self.tokenizer, relation_num)
        if self.args.load_dir != '':
            self.load(self.args.load_dir)

    def get_task_name(self):
        str_template = [str(x) for x in self.args.template]
        str_template = '.'.join(str_template)
        names = [self.args.model_name,
                 "template_{}".format(str_template),
                 "seed_{}".format(self.args.seed)]
        return "_".join(names)

    def get_save_path(self):
        return join(self.args.out_dir, self.args.model_name, 'search', self.get_task_name())

    def get_checkpoint(self, epoch_idx, dev_f1, test_f1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_f1 * 100, 4),
                                                          round(test_f1 * 100, 4))
        return {'model_state_dict': self.model.state_dict(),
                'ckpt_name': ckpt_name,
                'dev_f1': dev_f1,
                'test_f1': test_f1}

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        print("# Checkpoint {} saved.".format(ckpt_name))

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train(self):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        params = [{'params': self.model.prompt_encoder.parameters()}]
        if self.args.use_lm_finetune:
            params.append({'params': self.model.model.parameters(), 'lr': self.args.lm_lr})
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        for epoch_idx in range(self.args.max_epoch):
            # run training
            pbar = tqdm(self.train_loader)
            for batch_idx, batch in enumerate(pbar):
                self.model.train()
                loss, acc, _ = self.model.forward_classification(batch[0], batch[1], batch[2])
                pbar.set_description(f"Loss {float(loss.mean()):.6g}, acc {acc:.4g}")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # check early stopping
                if batch_idx % self.args.valid_step == 0:
                    # Triple Classification
                    dev_results, test_results = evaluate_classification_using_classification(self, epoch_idx)
                    
                    # Link Prediction
                    if self.args.link_prediction and not (batch_idx == 0 and epoch_idx == 0):
                        evaluate_link_prediction_using_classification(self, epoch_idx, batch_idx, output_scores=False)

                    # Early stop and save
                    if dev_results >= best_dev:
                        best_ckpt = self.get_checkpoint(epoch_idx, dev_results, test_results)
                        early_stop = 0
                        best_dev = dev_results
                    else:
                        early_stop += 1
                        if early_stop >= self.args.early_stop:
                            self.save(best_ckpt)
                            print("Early stopping at epoch {}.".format(epoch_idx))
                            print("FINISH_TRAIN...")
                            return best_ckpt

                    sys.stdout.flush()
            my_lr_scheduler.step()
        self.save(best_ckpt)

        return best_ckpt


def main():
    args = construct_generation_args()
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    print(args.model_name)
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
