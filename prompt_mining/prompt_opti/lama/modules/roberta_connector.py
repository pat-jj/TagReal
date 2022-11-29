# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from fairseq.models.roberta import RobertaModel
from fairseq import utils
from fairseq.tokenizer import tokenize_line
import torch
import torch.nn.functional as F
from lama.modules.base_connector import *
import numpy as np
from copy import deepcopy


class RobertaVocab(object):
    def __init__(self, roberta):
        self.roberta = roberta

    def __getitem__(self, arg):
        value = ""
        try:
            predicted_token_bpe = self.roberta.task.source_dictionary.string([arg])
            if (
                predicted_token_bpe.strip() == ROBERTA_MASK
                or predicted_token_bpe.strip() == ROBERTA_START_SENTENCE
            ):
                value = predicted_token_bpe.strip()
            else:
                value = self.roberta.bpe.decode(str(predicted_token_bpe)).strip()
        except Exception as e:
            print(arg)
            print(predicted_token_bpe)
            print(value)
            print("Exception {} for input {}".format(e, arg))
        return value


class Roberta(Base_Connector):
    def __init__(self, args):
        super().__init__()
        roberta_model_dir = args.roberta_model_dir
        roberta_model_name = args.roberta_model_name
        roberta_vocab_name = args.roberta_vocab_name
        self.dict_file = "{}/{}".format(roberta_model_dir, roberta_vocab_name)
        self.roberta_model = RobertaModel.from_pretrained(
            roberta_model_dir, checkpoint_file=roberta_model_name
        )
        self.bpe = self.roberta_model.bpe
        self.task = self.roberta_model.task
        self._build_vocab()
        self._init_inverse_vocab()
        self.max_sentence_length = args.max_sentence_length

    @property
    def model(self):
        return self.roberta_model

    @property
    def mask_token(self):
        return ROBERTA_MASK

    def tokenize(self, text: str, add_start: bool = True):
        masked_text = text.replace(MASK, ROBERTA_MASK)
        text_spans = masked_text.split(ROBERTA_MASK)
        text_spans_bpe = ' {0} '.format(ROBERTA_MASK).join(
            [self.bpe.encode(text_span.rstrip())for text_span in text_spans]).strip()
        if add_start:
            text_spans_bpe = ROBERTA_START_SENTENCE + ' ' + text_spans_bpe
        return tokenize_line(text_spans_bpe)

    def _cuda(self):
        self.model.cuda()

    def _build_vocab(self):
        self.vocab = []
        for key in range(ROBERTA_VOCAB_SIZE):
            predicted_token_bpe = self.task.source_dictionary.string([key])
            try:
                value = self.bpe.decode(predicted_token_bpe)

                if value[0] == " ":  # if the token starts with a whitespace
                    value = value.strip()
                else:
                    # this is subword information
                    value = "_{}_".format(value)

                if value in self.vocab:
                    # print("WARNING: token '{}' is already in the vocab".format(value))
                    value = "{}_{}".format(value, key)

                self.vocab.append(value)

            except Exception as e:
                self.vocab.append(predicted_token_bpe.strip())

    def get_id(self, input_string):
        # Roberta predicts ' London' and not 'London'
        string = " " + str(input_string).strip()
        text_spans_bpe = self.bpe.encode(string.rstrip())
        tokens = self.task.source_dictionary.encode_line(text_spans_bpe, append_eos=False, add_if_not_exist=False)
        return tokens.long().cpu().numpy().tolist()

    def tokenize_and_toid(self, text: str, add_start: bool = True):
       return self.task.source_dictionary.encode_line(
           ' '.join(self.tokenize(text, add_start=add_start)), append_eos=True, add_if_not_exist=False).long()

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True, relation_mask=None):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tensor_list = []
        relation_mask_tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        for sid, masked_inputs_list in enumerate(sentences_list):

            tokens_list = []
            rel_mask_list = []

            for idx, masked_input in enumerate(masked_inputs_list):
                if relation_mask is None:
                    tokens_list.append(self.tokenize_and_toid(masked_input, add_start=(idx == 0)))
                else:
                    tokens_list.append(self.task.source_dictionary.encode_line(' '.join(masked_input), append_eos=True, add_if_not_exist=False))
                    rel_mask_list.append(torch.tensor(relation_mask[sid][idx] + [0]).long())  # eos

            tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.cpu().numpy())
            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.task.mask_idx).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])

            if relation_mask is not None:
                relation_mask_tensor_list.append(torch.cat(rel_mask_list)[: self.max_sentence_length])

        pad_id = self.task.source_dictionary.pad()
        tokens_list = []
        for tokens in tensor_list:
            pad_lenght = max_len - len(tokens)
            if pad_lenght > 0:
                pad_tensor = torch.full([pad_lenght], pad_id, dtype=torch.long)
                tokens = torch.cat((tokens, pad_tensor))
            tokens_list.append(tokens)

        raw_tokens = torch.stack(tokens_list)
        if relation_mask is not None:
            rel_mask_tensor = torch.nn.utils.rnn.pad_sequence(relation_mask_tensor_list, batch_first=True, padding_value=0)

        if relation_mask is not None:
            batch_tokens = raw_tokens * rel_mask_tensor
        else:
            batch_tokens = raw_tokens

        with torch.no_grad():
            # with utils.eval(self.model.model):
            self.model.eval()
            self.model.model.eval()
            logits, extra = self.model.model(
                batch_tokens.to(device=self._model_device),
                features_only=False,
                return_all_hiddens=False,
            )
            log_probs = logits.log_softmax(dim=-1)

        mask_tensor = (batch_tokens == self.task.mask_idx).long() if relation_mask is None else rel_mask_tensor

        return log_probs.cpu(), output_tokens_list, masked_indices_list, raw_tokens, mask_tensor

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # TBA
        return None

    def refine_cloze(self, sentence_list: List[str], try_cuda: bool = False, batch_size: int = 32, beam_size: int = 1,
                     max_try: int = 10):
        tokenized_sent_list = []
        start_ind = []  # inclusive
        end_ind = []  # exclusive
        for sent in sentence_list:
            tokenized_sent = []
            ind = 0
            for w in re.split(' ', sent):
                if not w:
                    continue
                if w == '[':
                    start_ind.append(ind)
                elif w == ']':
                    end_ind.append(ind)
                else:
                    tokenized_sent.append(w)
                    ind += 1
            tokenized_sent_list.append(tokenized_sent)
        cloze_len = np.unique(np.array(end_ind) - np.array(start_ind))
        assert len(cloze_len) == 1, 'cloze do not have the same length'
        cloze_len = cloze_len[0]

        def mask_token_at(sent: List[str], pos: int):
            new_sent = deepcopy(sent)
            new_sent[pos] = self.mask_token   # replace the token at pos with a mask
            return new_sent

        if try_cuda:
            self.try_cuda()

        try_num = 0
        while True:
            modify_num = 0
            for i in range(cloze_len):
                # SHAPE: (one token len, vocab_size)
                probs_sum = 0

                for batch_ind in range(0, len(tokenized_sent_list), batch_size):
                    cur_sent_list = tokenized_sent_list[batch_ind:batch_ind + batch_size]
                    cur_start_ind = start_ind[batch_ind:batch_ind + batch_size]
                    new_sent_list = [' '.join(mask_token_at(ts, cur_start_ind[sid] + i)) for sid, ts in enumerate(cur_sent_list)]

                    batch_tokens = [self.tokenize_and_toid(s) for s in new_sent_list]
                    batch_tokens = torch.nn.utils.rnn.pad_sequence(
                        batch_tokens, batch_first=True, padding_value=self.task.source_dictionary.pad()).to(device=self._model_device)
                    bracket = (batch_tokens == self.task.mask_idx).long().to(device=self._model_device)

                    self.model.eval()
                    self.model.model.eval()
                    log_probs, extra = self.model.model(
                        batch_tokens,
                        features_only=False,
                        return_all_hiddens=False,
                    )
                    probs = F.softmax(log_probs, -1)
                    # SHAPE: (batch_size, one token len, vocab_size)
                    probs = probs.masked_select(bracket.eq(1).unsqueeze(-1)).view(probs.size(0), -1, probs.size(-1))
                    probs_sum = probs_sum + probs.sum(0).detach()

                # SHAPE: (one token len)
                replace = probs_sum.max(-1)[1]
                replace = self.model.decode(replace.cpu())
                if type(replace) is not list:
                    replace = [replace]
                if len(replace) > 1:
                    '''
                    dst[dst == -1] = 0
                    print(new_sent_list)
                    print(self.tokenizer.convert_ids_to_tokens(src[0].cpu().numpy()))
                    print(self.tokenizer.convert_ids_to_tokens(dst[0].cpu().numpy()))
                    print(replace, tokenized_sent_list[0][start_ind[0] + i])
                    input()
                    '''
                    replace = ''.join(replace)
                    # raise Exception('one token splitted into multiple pieces')
                else:
                    # assert len(replace) == 1, 'one token splitted into multiple pieces'
                    replace = replace[0].strip()
                old = tokenized_sent_list[0][start_ind[0] + i]

                # replace tokens
                if old != replace:
                    modify_num += 1
                    print('modify {} to {}'.format(old, replace))
                    for sid, ts in enumerate(tokenized_sent_list):
                        ts[start_ind[sid] + i] = replace

            if modify_num == 0:
                break
            try_num += 1
            if try_num >= max_try:
                print('max try reached')
                break

        new_cloze = [tokenized_sent_list[0][start_ind[0] + i] for i in range(cloze_len)]
        return new_cloze
