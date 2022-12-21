from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os
from tqdm import tqdm
    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.1, model='tucker', dataset='infer', filt=False):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.model = model
        self.cuda = cuda
        self.dataset = dataset
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        self.filt = filt
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = torch.zeros(len(batch), len(d.entities), device='cuda')
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        return np.array(batch), targets

    
    def get_train_kge_neg(self, model, data):
        ft = open('output/{}.{}.{}.kge_neg.txt'.format(self.dataset, self.model, self.ent_vec_dim), 'w')
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            e1_idx_cpu = e1_idx.tolist()
            r_idx_cpu = r_idx.tolist()
            e2_idx_cpu = e2_idx.tolist()
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                # predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()

            for j in range(data_batch.shape[0]):
                for k in range(100):
                    ft.write('{}\t{}\t{}\n'.format(self.idxs_entity[int(e1_idx_cpu[j])], self.idxs_relation[int(r_idx_cpu[j])], self.idxs_entity[int(sort_idxs[j][k])]))

        ft.close()

    def evaluate(self, model, data, out_lp_constraints=False):
        f = open('output/{}.{}.{}.txt'.format(self.dataset, self.model, self.ent_vec_dim), 'w')
        if out_lp_constraints:
            ft = open('output/{}.{}.{}.test.link_prediction.txt'.format(self.dataset, self.model, self.ent_vec_dim), 'w')
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
        self.seen_data = d.train_data + d.valid_data +d.test_data
        self.seen_data = set(map(tuple, self.seen_data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            e1_idx_cpu = e1_idx.tolist()
            r_idx_cpu = r_idx.tolist()
            e2_idx_cpu = e2_idx.tolist()
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                # if self.filt:
                    # filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                    # target_value = predictions[j,e2_idx[j]].item()
                    # predictions[j, filt] = 0.0
                    # predictions[j, e2_idx[j]] = target_value

                # else:
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()

            if out_lp_constraints:
                for j in range(data_batch.shape[0]):
                    for k in range(500):
                        ft.write('{}\t{}\t{}\n'.format(self.idxs_entity[int(e1_idx_cpu[j])], self.idxs_relation[int(r_idx_cpu[j])], self.idxs_entity[int(sort_idxs[j][k])]))
                    # ft.write('SPLIT\n')

            for j in range(data_batch.shape[0]):
                
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]

                # if self.filt:

                #     for h in range(10):
                #         e1_in = self.idxs_entity[e1_idx[j].item()]
                #         r_in = self.idxs_relation[r_idx[j].item()]
                #         e2_pred = self.idxs_entity[sort_idxs[j][h]]
                #         # print((e1_in, r_in, e2_pred))
                #         if (e1_in, r_in, e2_pred) in self.seen_data:
                #             # take the highest rank
                #             rank = h
                #             break

                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                f.write('{}\t{}\t{}\t{}\n'.format(self.idxs_entity[int(e1_idx_cpu[j])], self.idxs_relation[int(r_idx_cpu[j])], self.idxs_entity[int(e2_idx_cpu[j])], rank + 1))

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @5: {0}'.format(np.mean(hits[4])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))
        f.close()
        if out_lp_constraints:
            ft.close()

    def train_and_eval(self):
        print("Training the {} model...".format(self.model))
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        self.idxs_entity = {i:d.entities[i] for i in range(len(d.entities))}
        self.idxs_relation = {i:d.relations[i] for i in range(len(d.relations))}

        print(f"Number of Entity: {len(self.entity_idxs)}")

        if self.model == 'tucker':
            model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif self.model == 'hyper':
            model = HypER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif self.model == 'conve':
            model = ConvE(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        elif self.model == 'roberta':
            model = RoBERTa(d, self.ent_vec_dim, self.idxs_entity, self.idxs_relation, f'data/{self.dataset}')
        elif self.model == 'T5':
            model = T5(d, self.ent_vec_dim, self.idxs_entity, self.idxs_relation, f'data/{self.dataset}')
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(0, self.num_iterations):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in tqdm(range(0, len(er_vocab_pairs), self.batch_size)):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print('Epoch: {}, Time: {}s, Loss: {}'.format(it, time.time()-start_train, np.mean(losses)))
            model.eval()
            with torch.no_grad():
                if it % 50 == 0 and it != 0:
                    print("Test:")
                    self.evaluate(model, d.test_data, out_lp_constraints=False)
                if it == self.num_iterations - 1:
                    self.evaluate(model, d.test_data, out_lp_constraints=True)
                    self.get_train_kge_neg(model, d.train_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--model", type=str, default='tucker', nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--gpu", type=int, default=2, nargs="?",
                help="Relation embedding dimensionality.") 
    parser.add_argument("--filt", type=bool, default=False, help="Apply filter?")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 5583
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing, model=args.model, dataset=dataset, filt=args.filt)
    experiment.train_and_eval()
                

