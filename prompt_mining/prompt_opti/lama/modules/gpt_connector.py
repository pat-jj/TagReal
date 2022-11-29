# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List
import re
import torch.nn as nn
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import numpy as np
from copy import deepcopy
from lama.modules.base_connector import *


class GPT(Base_Connector):

    EMB_DIM = 40478  # TODO: identify the embedding using its dimension

    def __init__(self, args):
        super().__init__()

        if args.gpt_model_dir is not None:
            # load bert model from file
            gpt_model_name = str(args.gpt_model_dir) + "/"
            dict_file = gpt_model_name
            print("loading Open AI GPT model from {}".format(gpt_model_name))
        else:
            # load GPT model from huggingface cache
            gpt_model_name = args.gpt_model_name
            dict_file = gpt_model_name

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(dict_file)

        # GPT uses different way to represent BPE then BERT. Namely, the
        # final suffixes are indicated with </w> suffix, while pieces that must
        # be followed are written as is. In BERT the prefixes are written as is
        # while the parts that must follow (not be followed!) have '##' prefix.
        # There is no one-to-one coversion. But at least we may make pieces that
        # may form a full word look the same.
        # Note that we should be very careful now,
        # tokenizer.convert_tokens_to_ids won't work with our vocabulary.
        def convert_word(word):
            if word == OPENAI_UNK:
                return word
            if word == '\n</w>':
                # Redefine symbol EOS to improve visualization.
                return OPENAI_EOS
            return word[:-4] if word.endswith('</w>') else f'{word}##'

        _, gpt_vocab = zip(*sorted(self.tokenizer.decoder.items()))
        self.vocab = [convert_word(word) for word in gpt_vocab]
        self._init_inverse_vocab()

        # Get UNK symbol as it's written in the origin GPT vocab.
        unk_index = self.inverse_vocab[OPENAI_UNK]
        self.unk_symbol = self.tokenizer.decoder[unk_index]

        # Load pre-trained model (weights)
        self.gpt_model = OpenAIGPTLMHeadModel.from_pretrained(gpt_model_name)
        self.gpt_model.eval()
        print(self.gpt_model.config)

        # Sanity check.
        assert len(self.vocab) == self.gpt_model.config.vocab_size
        assert 0 == self.gpt_model.config.n_special

        self.eos_id = self.inverse_vocab[OPENAI_EOS]
        self.model_vocab = self.vocab

    @property
    def model(self):
        return self.gpt_model

    def _cuda(self):
        self.gpt_model.cuda()

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # indexed_string = self.convert_ids(indexed_string)
        return indexed_string

    def __get_input_tensors(self, sentence_list):
        """Concatenates, tokenize and converts a sentences to model inputs.

        Args:
            sentence_list: A list of strings. The string may contain a special
            [MASK] token.

        Returns:
            A tuple (src_tensor, dst_tensor, masked_indices, tokenized_text).
                src_tensor: torch.LongTensor with shape (seq_len), the input to
                    the new without the last symbol and with EOS prepended.
                dst_tensor: torch.LongTensor with shape (seq_len).
                masked_indices: A list of indices of [MASK] in dst_tensor.
                tokenized_text: A list of token string.
            """
        # Split the sentence by [MASK] and tokenize the chunks independently.
        tokenized_text = []
        masked_indices = []
        for sentence_idx, sentence in enumerate(sentence_list):
            if sentence_idx > 0:
                tokenized_text.append(OPENAI_EOS)
            for chunk_idx, chunk in enumerate(sentence.split('[MASK]')):
                if chunk_idx > 0:
                    masked_indices.append(len(tokenized_text))
                    tokenized_text.append(self.unk_symbol)
                chunk = chunk.strip()
                if chunk:
                    tokenized_text.extend(self.tokenizer.tokenize(chunk))

        full_indexed_tokens = [
            self.eos_id
        ] + self.tokenizer.convert_tokens_to_ids(tokenized_text)
        full_tokens_tensor = torch.tensor(full_indexed_tokens)
        src_tensor = full_tokens_tensor[:-1]
        dst_tensor = full_tokens_tensor[1:]

        return src_tensor, dst_tensor, masked_indices, tokenized_text

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()
        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        # The model uses shared embedding space for tokens and positions. More
        # precisely, the first len(vocab) indidices are reseved for words, the
        # last n_special symbols are reserved for special symbols and the rest
        # is used for positions. Softmax and embedding matrices are shared and
        # as result some of output "symbols" correspond to positions. To fix
        # that we have to manually remove logits for positions.
        with torch.no_grad():
            logits = self.gpt_model(src_tensor_batch.to(self._model_device))
            logits = logits[..., :self.gpt_model.config.vocab_size]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu()

        token_ids_list = [
            np.array(dst_tensor.numpy()) for dst_tensor in dst_tensor_list
        ]

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        if try_cuda:
            self.try_cuda()

        src_tensor_list, dst_tensor_list, masked_indices_list, _ = zip(*[
            self.__get_input_tensors(sentences) for sentences in sentences_list
        ])

        src_tensor_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensor_list, batch_first=True)

        with torch.no_grad():
            output = self.gpt_model.transformer(src_tensor_batch.to(self._model_device))

        # TODO
        sentence_lengths = None
        tokenized_text_list = None

        # As we only return the last layer, [] to have the same format as other models
        return [output], sentence_lengths, tokenized_text_list

    def __get_one_tensor_with_mask(self, sentence: str):
        # Split the sentence by brackets and tokenize the chunks independently.
        tokenized_text, bracket, unbracket = self.tokenizer_with_mask(sentence)

        token_ids = [self.eos_id] + self.tokenizer.convert_tokens_to_ids(tokenized_text)
        bracket = [0] + bracket
        unbracket = [0] + unbracket
        src_tensor = torch.tensor(token_ids)
        dst_tensor = torch.tensor(token_ids)
        token_ids = torch.tensor(token_ids)
        bracket_tensor = torch.tensor(bracket)
        unbracket_tensor = torch.tensor(unbracket)
        dst_tensor = dst_tensor * bracket_tensor
        dst_tensor[dst_tensor == 0] = -1

        return src_tensor, dst_tensor, bracket_tensor, unbracket_tensor, token_ids

    def __get_batch_tensors_with_mask(self, sentence_list: List[str]):
        src, dst, bracket, unbracket, token_ids = \
            zip(*[self.__get_one_tensor_with_mask(sentence) for sentence in sentence_list])

        # SHAPE: (batch_size, seq_len)
        src_batch = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)
        dst_batch = torch.nn.utils.rnn.pad_sequence(dst, padding_value=-1, batch_first=True)
        bracket_batch = torch.nn.utils.rnn.pad_sequence(bracket, batch_first=True)
        unbracket_batch = torch.nn.utils.rnn.pad_sequence(unbracket, batch_first=True)
        token_ids_batch = torch.nn.utils.rnn.pad_sequence(token_ids, padding_value=-1, batch_first=True)

        return src_batch, dst_batch, bracket_batch, unbracket_batch, token_ids_batch

    def get_rc_loss(self, sentence_list: List[str], try_cuda: bool = False):
        src, dst, bracket, unbracket, token_ids = self.__get_batch_tensors_with_mask(sentence_list)

        if try_cuda:
            self.try_cuda()
            src = src.cuda()
            dst = dst.cuda()
            bracket = bracket.cuda()
            unbracket = unbracket.cuda()
            token_ids = token_ids.cuda()

        self.model.eval()
        loss = self.model(src, lm_labels=dst)
        return loss, token_ids, bracket, unbracket

    def fill_cloze(self, sentence_list: List[str], try_cuda: bool = False, beam_size: int = 5, bbatch_size: int = 16):
        acc_token_li, acc_sent_li = [], []
        for sentence in sentence_list:
            src, dst, bracket, unbracket, _ = self.__get_one_tensor_with_mask(sentence)
            score = torch.tensor([0.0])

            if try_cuda:
                self.try_cuda()
                src = src.cuda()
                dst = dst.cuda()
                score = score.cuda()

            self.model.eval()
            # SHAPE: (beam_size, seq_len + 1)
            src_ = src.clone().unsqueeze(0)
            bracket_ind = torch.arange(bracket.size(0) - 1)[bracket[1:].eq(1)]  # remove first token
            for ind in bracket_ind:
                # get scores
                # SHAPE: (beam_size, seq_len + 1, vocab_size)
                logits = torch.cat([self.model(src__[:, :ind + 1]).detach() for src__ in torch.split(src_, bbatch_size, dim=0)], dim=0)
                # SHAPE: (beam_size, vocab_size)
                logits_cur = logits[:, ind, :]
                logits_cum = logits_cur + score.unsqueeze(-1)
                score, indices = torch.topk(logits_cum.view(-1), k=beam_size)

                # new src
                src_np = src_.cpu().numpy()
                new_src_np = []
                vocab_size = logits.size(-1)
                for index in indices.cpu().numpy():
                    c, w = index // vocab_size, index % vocab_size
                    n = deepcopy(src_np[c])
                    n[ind + 1] = w
                    new_src_np.append(n)
                src_ = torch.tensor(new_src_np)
                src_ = src_.cuda() if try_cuda else src_
                '''
                print(src_[:, :ind + 2])
                print(np.array(self.tokenizer.convert_ids_to_tokens(src_[:, :ind + 2].cpu().numpy().reshape(-1))).reshape(src_.size(0), -1))
                input()
                '''

            bracket_back = bracket.clone()
            back = False
            for i, b in zip(range(bracket.size(0)), bracket):
                if b != 0 or back:
                    bracket_back[i] = 1
                    back = True
                else:
                    bracket_back[i] = 0
            bracket_back = bracket_back.unsqueeze(0)

            if try_cuda:
                bracket = bracket.cuda()
                bracket_back = bracket_back.cuda()

            dst_ = src_.clone() * bracket_back
            dst_[dst_ == 0] = -1
            logits = torch.cat([self.model(src__).detach() for src__ in torch.split(src_, bbatch_size, dim=0)], dim=0)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_dst = dst_[:, 1:].contiguous().view(-1)
            # SHPAE: (beam_size, seq_len)
            loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(shift_logits, shift_dst).view(logits.size(0), -1)
            _, ind = loss.sum(-1).min(0)
            logits_argmax = src_[ind]

            acc_token = (logits_argmax.eq(dst).long() * bracket).sum().float() / (bracket.sum() + 1e-10).float()
            acc_sent = (logits_argmax.eq(dst).long() * bracket).sum().eq(bracket.sum()).float()
            acc_token_li.append(acc_token.item())
            acc_sent_li.append(acc_sent.item())
            '''
            print(sentence)
            print(self.tokenizer.convert_ids_to_tokens(src[bracket == 1].cpu().numpy()))
            print(self.tokenizer.convert_ids_to_tokens(logits_argmax[bracket == 1].cpu().numpy()))
            print(acc_token.item(), acc_sent.item())
            input()
            '''
        return np.mean(acc_token_li), np.mean(acc_sent_li)
