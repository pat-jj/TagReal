from torch.utils.data import DataLoader
from data_utils.dataset import *
from os.path import join, abspath, dirname

def get_dataloader(args, tokenizer):
    basic_data = BasicDataWiki(args, tokenizer)

    neg_file_kge = join(args.data_dir, f'train_neg_kge_{args.keg_neg}.txt')
    if args.random_neg_ratio == 1.0:
        neg_file_kge = None

    train_set = KEDatasetWiki(
        join(args.data_dir, 'train.txt'), 
        join(args.data_dir, 'train_neg_rand.txt'),
        basic_data,
        neg_file_kge=neg_file_kge,
        pos_K=args.pos_K,
        neg_K=args.neg_K,
        random_neg_ratio=args.random_neg_ratio
    )
    test_set = KEDatasetWiki(
        join(args.data_dir, 'test_pos.txt'), 
        join(args.data_dir, 'test_neg.txt'), 
        basic_data
    )
    dev_set = KEDatasetWiki(
        join(args.data_dir, 'valid_pos.txt'), 
        join(args.data_dir, 'valid_neg.txt'), 
        basic_data
    )

    if args.test_open:
        o_test_set = KEDatasetWiki(
            join(args.data_dir, 'o_test_pos.txt'), 
            join(args.data_dir, 'o_test_neg.txt'), 
            basic_data
        )
    if args.link_prediction:
        link_dataset_tail = KEDatasetWikiInfer(
            join(args.data_dir, 'link_prediction_tail.txt'), 
            basic_data, 
            args.recall_k
        )
        link_dataset_head = KEDatasetWikiInfer(
            join(args.data_dir, 'link_prediction_head.txt'), 
            basic_data, 
            args.recall_k
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    ch_test_loader, oh_test_loader = None, None

    if args.test_open:
        o_test_loader = DataLoader(o_test_set, batch_size=args.batch_size)
    else:
        o_test_loader = None

    if args.link_prediction:
        link_loader_tail = DataLoader(link_dataset_tail, batch_size=args.batch_size)
        link_loader_head = DataLoader(link_dataset_head, batch_size=args.batch_size)
    else:
        link_loader_tail = None
        link_loader_head = None
        link_dataset_tail = None
        link_dataset_head = None
    return train_loader, dev_loader, test_loader, ch_test_loader, oh_test_loader, o_test_loader, link_loader_head, link_loader_tail, len(basic_data.relation2idx), link_dataset_head, link_dataset_tail