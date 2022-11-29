# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np


def __print_top_k(value_max_probs, index_max_probs, vocab, mask_topk, index_list, max_printouts = 10):
    result = []
    msg = "\n| Top{} predictions\n".format(max_printouts)
    for i in range(mask_topk):
        filtered_idx = index_max_probs[i].item()

        if index_list is not None:
            # the softmax layer has been filtered using the vocab_subset
            # the original idx should be retrieved
            idx = index_list[filtered_idx]
        else:
            idx = filtered_idx

        log_prob = value_max_probs[i].item()
        word_form = vocab[idx]

        if i < max_printouts:
            msg += "{:<8d}{:<20s}{:<12.3f}\n".format(
                i,
                word_form,
                log_prob
            )
        element = {'i' : i, 'token_idx': idx, 'log_prob': log_prob, 'token_word_form': word_form}
        result.append(element)
    return result, msg


def get_ranking(log_probs, vocab, label_index = None, index_list = None, topk = 1000, P_AT = 10, print_generation=True):
    experiment_result = {}

    value_max_probs, index_max_probs = torch.topk(input=log_probs, k=topk, dim=0)
    index_max_probs = index_max_probs.numpy().astype(int)
    value_max_probs = value_max_probs.detach().numpy()

    result_masked_topk, return_msg = __print_top_k(value_max_probs, index_max_probs, vocab, topk, index_list)
    experiment_result['topk'] = result_masked_topk

    if print_generation:
        print(return_msg)

    MRR = 0.
    P_AT_X = 0.
    P_AT_1 = 0.
    PERPLEXITY = None

    if label_index is not None:

        # check if the labe_index should be converted to the vocab subset
        if index_list is not None:
            label_index = index_list.index(label_index)

        query = torch.full(value_max_probs.shape, label_index, dtype=torch.long).numpy().astype(int)
        ranking_position = (index_max_probs == query).nonzero()

        # LABEL PERPLEXITY
        tokens = torch.from_numpy(np.asarray(label_index))
        label_perplexity = log_probs.gather(
            dim=0,
            index=tokens,
        )
        PERPLEXITY = label_perplexity.item()

        if len(ranking_position) > 0 and ranking_position[0].shape[0] != 0:
            rank = ranking_position[0][0] + 1

            # print("rank: {}".format(rank))

            if rank >= 0:
                MRR = (1 / rank)
            if rank >= 0 and rank <= P_AT:
                P_AT_X = 1.
            if rank == 1:
                P_AT_1 = 1.

    experiment_result["MRR"] = MRR
    experiment_result["P_AT_X"] = P_AT_X
    experiment_result["P_AT_1"] = P_AT_1
    experiment_result["PERPLEXITY"] = PERPLEXITY
    #
    # print("MRR: {}".format(experiment_result["MRR"]))
    # print("P_AT_X: {}".format(experiment_result["P_AT_X"]))
    # print("P_AT_1: {}".format(experiment_result["P_AT_1"]))
    # print("PERPLEXITY: {}".format(experiment_result["PERPLEXITY"]))

    return MRR, P_AT_X, experiment_result, return_msg


def get_ranking_select(log_probs, masked_indices, vocab, label_index = None, index_list = None, topk = 1000, P_AT = 10, print_generation=True):
    # score only first mask
    masked_indices = masked_indices[:1]

    masked_index = masked_indices[0]
    log_probs = log_probs[masked_index]

    return get_ranking(log_probs, vocab, label_index=label_index, index_list=index_list, topk=topk, P_AT=P_AT, print_generation=print_generation)


def analyze_prob(log_prob: torch.FloatTensor,  # SHAPE: (batch_size, num_temp, vocab_size)
                 label_index: torch.LongTensor,  # SHAPE: (batch_size)
                 output: bool = False,
                 method: str = 'all'
                 ):
    topk = 5
    show_num = 5
    vocab_size = log_prob.size(-1)
    prob = log_prob.exp()
    # SHAPE: (batch_size, num_temp, topk)
    prob_top, prob_top_ind = torch.topk(input=prob, k=topk, dim=-1)
    # SHAPE: (batch_size, num_temp)
    correct_mask = prob_top_ind[:, :, 0].eq(label_index.view(-1, 1))
    # SHAPE: (batch_size, num_temp)
    prob_gap = prob_top[:, :, 0] - prob_top[:, :, 1]
    prob_abs = prob_top[:, :, 0]

    c_prob_top = prob_top.masked_select(correct_mask.unsqueeze(-1)).view(-1, topk)
    inc_prob_top = prob_top.masked_select(~correct_mask.unsqueeze(-1)).view(-1, topk)

    if method == 'all':
        ## overall statistics
        # SHAPE: (None)
        c_prob_gap = prob_gap.masked_select(correct_mask)
        c_prob_abs = prob_abs.masked_select(correct_mask)
        # SHAPE: (None, topk)
        inc_prob_gap = prob_gap.masked_select(~correct_mask)
        inc_prob_abs = prob_abs.masked_select(~correct_mask)
        num_c, num_inc = c_prob_gap.size(0), inc_prob_gap.size(0)

        c_gap = c_prob_gap.sum().item()
        inc_gap = inc_prob_gap.sum().item()
        c_abs = c_prob_abs.sum().item()
        inc_abs = inc_prob_abs.sum().item()

    elif method == 'sample':
        ## sample-wise statistics
        correct_mask = correct_mask.float()
        incorrect_mask = 1 - correct_mask
        # only consider samples with both correct and incorrect templates
        correct_mask = correct_mask * incorrect_mask.max(-1, keepdim=True)[0]
        incorrect_mask = incorrect_mask * correct_mask.max(-1, keepdim=True)[0]

        c_gap = (prob_gap * correct_mask).max(-1)[0].sum().item()
        c_abs = (prob_abs * correct_mask).max(-1)[0].sum().item()
        inc_gap = (prob_gap * incorrect_mask).max(-1)[0].sum().item()
        inc_abs = (prob_abs * incorrect_mask).max(-1)[0].sum().item()
        num_c = correct_mask.max(-1)[0].sum()
        num_inc = incorrect_mask.max(-1)[0].sum()

    else:
        raise Exception

    if output:
        print('#correct temp {}, #incorrect temp {}'.format(num_c, num_inc))
        print('correct')
        print(c_prob_top[np.random.choice(num_c, min(show_num, num_c), replace=False)])
        print('incorrect')
        print(inc_prob_top[np.random.choice(num_inc, min(show_num, num_inc), replace=False)])

    return (c_gap, c_abs, num_c), (inc_gap, inc_abs, num_inc)
