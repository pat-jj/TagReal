# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import pytorch_pretrained_bert.tokenization as btok
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BasicTokenizer, BertModel
import numpy as np
from copy import deepcopy
from lama.modules.base_connector import *
import torch.nn.functional as F


class CustomBaseTokenizer(BasicTokenizer):

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = btok.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:

            # pass MASK forward
            if MASK in token:
                split_tokens.append(MASK)
                if token != MASK:
                    remaining_chars = token.replace(MASK,"").strip()
                    if remaining_chars:
                        split_tokens.append(remaining_chars)
                continue

            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = btok.whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


class Bert(Base_Connector):

    EMB_DIM = 28996  # TODO: identify the embedding using its dimension

    def __init__(self, args, vocab_subset = None):
        super().__init__()

        bert_model_name = args.bert_model_name
        dict_file = bert_model_name

        if args.bert_model_dir is not None:
            # load bert model from file
            bert_model_name = str(args.bert_model_dir) + "/"
            dict_file = bert_model_name+args.bert_vocab_name
            self.dict_file = dict_file
            print("loading BERT model from {}".format(bert_model_name))
        else:
            # load bert model from huggingface cache
            pass

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in bert_model_name:
            do_lower_case=True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(dict_file)

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Add custom tokenizer to avoid splitting the ['MASK'] token
        custom_basic_tokenizer = CustomBaseTokenizer(do_lower_case = do_lower_case)
        self.tokenizer.basic_tokenizer = custom_basic_tokenizer

        # Load pre-trained model (weights)
        # ... to get prediction/generation
        self.masked_bert_model = BertForMaskedLM.from_pretrained(bert_model_name)

        self.masked_bert_model.eval()

        # ... to get hidden states
        self.bert_model = self.masked_bert_model.bert

        self.pad_id = self.inverse_vocab[BERT_PAD]

        self.unk_index = self.inverse_vocab[BERT_UNK]

    @property
    def model(self):
        return self.masked_bert_model

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)



    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def __get_input_tensors_batch(self, sentences_list, relation_mask=None):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        masked_list_list = []
        max_tokens = 0
        for sid, sentences in enumerate(sentences_list):
            rm = relation_mask[sid] if relation_mask is not None else None
            tokens_tensor, segments_tensor, masked_indices, tokenized_text, masked_list = self.__get_input_tensors(sentences, relation_mask=rm)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            masked_list_list.append(masked_list)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
        # print(final_tokens_tensor)
        # print(final_segments_tensor)
        # print(final_attention_mask)
        # print(final_tokens_tensor.shape)
        # print(final_segments_tensor.shape)
        # print(final_attention_mask.shape)

        mask_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(m) for m in masked_list_list], batch_first=True, padding_value=0)

        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list, mask_tensor

    def __get_input_tensors(self, sentences, relation_mask=None):

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0]) if relation_mask is None else deepcopy(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(BERT_SEP)
        first_segment_id.append(0)

        if relation_mask is not None:
            all_relation_mask = deepcopy(relation_mask[0])
            all_relation_mask.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1]) if relation_mask is None else deepcopy(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(BERT_SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id

            if relation_mask is not None:
                all_relation_mask.extend(relation_mask[1])
                all_relation_mask.append(0)
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0,BERT_CLS)
        segments_ids.insert(0,0)
        if relation_mask is not None:
            all_relation_mask.insert(0, 0)

        # look for masked indices
        masked_indices = []
        masked_list = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == MASK:
                masked_indices.append(i)
                masked_list.append(0)
            else:
                masked_list.append(1)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        if relation_mask is not None:
            masked_list = all_relation_mask

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text, masked_list

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.masked_bert_model.cuda()

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True, relation_mask=None):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        raw_tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list, mask_tensor = \
            self.__get_input_tensors_batch(sentences_list, relation_mask=relation_mask)

        if relation_mask is not None:  # mask out relational phrase
            tokens_tensor = raw_tokens_tensor * (1 - mask_tensor) + mask_tensor * self.tokenizer.convert_tokens_to_ids([MASK])[0]
        else:
            tokens_tensor = raw_tokens_tensor

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            log_probs = F.log_softmax(logits, dim=-1).cpu()

        token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))

        return log_probs, token_ids_list, masked_indices_list, raw_tokens_tensor, mask_tensor

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_encoder_layers, _ = self.bert_model(
                tokens_tensor.to(self._model_device),
                segments_tensor.to(self._model_device))

        all_encoder_layers = [layer.cpu() for layer in all_encoder_layers]

        sentence_lengths = [len(x) for x in tokenized_text_list]

        # all_encoder_layers: a list of the full sequences of encoded-hidden-states at the end
        # of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
        # encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
        return all_encoder_layers, sentence_lengths, tokenized_text_list

    def __get_one_tensor_with_mask(self, sentence: str, mask_source: bool = True):
        tokenized_text, bracket_indices, unbracket_indices = self.tokenizer_with_mask(sentence)
        segment_id = np.zeros(len(tokenized_text), dtype=int).tolist()

        # add [SEP] token at the end
        tokenized_text.append(BERT_SEP)
        segment_id.append(0)
        bracket_indices.append(0)
        unbracket_indices.append(0)
        # add [CLS] token at the beginning
        tokenized_text.insert(0, BERT_CLS)
        segment_id.insert(0, 0)
        bracket_indices.insert(0, 0)
        unbracket_indices.insert(0, 0)

        src_tokens = deepcopy(tokenized_text)
        # mask out bracket
        if mask_source:
            for i, b in enumerate(bracket_indices):
                src_tokens[i] = MASK if b else src_tokens[i]

        # convert to ids
        src_tokens = self.tokenizer.convert_tokens_to_ids(src_tokens)
        dst_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        src_tensor = torch.tensor(src_tokens)
        dst_tensor = torch.tensor(dst_tokens)
        bracket_tensor = torch.tensor(bracket_indices)
        unbracket_tensor = torch.tensor(unbracket_indices)
        segments_tensors = torch.tensor(segment_id)
        dst_tokens = torch.tensor(dst_tokens)

        # mask out irrelevant positions
        dst_tensor = dst_tensor * bracket_tensor
        dst_tensor[dst_tensor == 0] = -1

        return src_tensor, dst_tensor, bracket_tensor, unbracket_tensor, segments_tensors, dst_tokens

    def __get_batch_tensors_with_mask(self, sentence_list: List[str]):
        src, dst, bracket, unbracket, segments, token_ids = \
            zip(*[self.__get_one_tensor_with_mask(sentence, mask_source=True) for sentence in sentence_list])
        seq_len = [s.size(0) for s in src]
        max_len = np.max(seq_len)

        src_batch = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=self.pad_id)
        dst_batch = torch.nn.utils.rnn.pad_sequence(dst, batch_first=True, padding_value=-1)
        bracket_batch = torch.nn.utils.rnn.pad_sequence(bracket, batch_first=True)
        unbracket_batch = torch.nn.utils.rnn.pad_sequence(unbracket, batch_first=True)
        segments_batch = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True)
        token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=-1)
        attention_mask = torch.arange(max_len).view(1, -1).long()
        attention_mask = (attention_mask < torch.tensor(seq_len).unsqueeze(-1)).long()

        return src_batch, dst_batch, bracket_batch, unbracket_batch, segments_batch, attention_mask, token_ids

    def get_rc_loss(self, sentence_list: List[str], try_cuda: bool = False):
        src, dst, bracket, unbracket, segments, attention_mask, token_ids = \
            self.__get_batch_tensors_with_mask(sentence_list)

        if try_cuda:
            self.try_cuda()
            src = src.cuda()
            dst = dst.cuda()
            bracket = bracket.cuda()
            unbracket = unbracket.cuda()
            segments = segments.cuda()
            attention_mask = attention_mask.cuda()
            token_ids = token_ids.cuda()

        self.model.train()
        loss = self.model(src,
                          token_type_ids=segments,
                          attention_mask=attention_mask,
                          masked_lm_labels=dst)
        return loss, token_ids, bracket, unbracket

    def refine_cloze(self, sentence_list: List[str], try_cuda: bool = False, batch_size: int = 32, beam_size: int = 1, max_try: int = 10):
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

        def bracket_token_at(sent: List[str], pos: int):
            new_sent = deepcopy(sent)
            # replace the token at pos with a single character to avoid it being splitted into multiple pieces
            new_sent[pos] = 'a'
            new_sent.insert(pos, '[')
            new_sent.insert(pos + 2, ']')
            return new_sent

        try_num = 0
        while True:
            modify_num = 0
            for i in range(cloze_len):
                # SHAPE: (one token len, vocab_size)
                probs_sum = 0

                for batch_ind in range(0, len(tokenized_sent_list), batch_size):
                    cur_sent_list = tokenized_sent_list[batch_ind:batch_ind + batch_size]
                    cur_start_ind = start_ind[batch_ind:batch_ind + batch_size]
                    new_sent_list = [' '.join(bracket_token_at(ts, cur_start_ind[sid] + i)) for sid, ts in enumerate(cur_sent_list)]
                    src, dst, bracket, unbracket, segments, attention_mask, _ = self.__get_batch_tensors_with_mask(new_sent_list)

                    if try_cuda:
                        self.try_cuda()
                        src = src.cuda()
                        dst = dst.cuda()
                        bracket = bracket.cuda()
                        unbracket = unbracket.cuda()
                        segments = segments.cuda()
                        attention_mask = attention_mask.cuda()

                    # run model
                    self.model.eval()
                    # SHAPE: (batch_size, seq_len, vocab_size)
                    probs = F.softmax(self.model(src, token_type_ids=segments, attention_mask=attention_mask), -1)
                    # SHAPE: (batch_size, one token len, vocab_size)
                    probs = probs.masked_select(bracket.eq(1).unsqueeze(-1)).view(probs.size(0), -1, probs.size(-1))
                    probs_sum = probs_sum + probs.sum(0).detach()

                # SHAPE: (one token len)
                replace = probs_sum.max(-1)[1]
                replace = self.tokenizer.convert_ids_to_tokens(replace.cpu().numpy())
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
                    #raise Exception('one token splitted into multiple pieces')
                else:
                    #assert len(replace) == 1, 'one token splitted into multiple pieces'
                    replace = replace[0]
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

    def fill_cloze(self, sentence_list: List[str], try_cuda: bool = False, beam_size: int = 1):
        src, dst, bracket, unbracket, segments, attention_mask, _ = self.__get_batch_tensors_with_mask(sentence_list)

        if try_cuda:
            self.try_cuda()
            src = src.cuda()
            dst = dst.cuda()
            bracket = bracket.cuda()
            unbracket = unbracket.cuda()
            segments = segments.cuda()
            attention_mask = attention_mask.cuda()

        self.model.eval()
        # SHAPE: (batch_size, seq_len, vocab_size)
        logits = self.model(src, token_type_ids=segments, attention_mask=attention_mask)
        logits_argmax = logits.max(-1)[1]

        acc_token = (logits_argmax.eq(dst).long() * bracket).sum().float() / (bracket.sum() + 1e-10).float()
        acc_sent = (logits_argmax.eq(dst).long() * bracket).sum(-1).eq(bracket.sum(-1)).sum().float() / bracket.size(0)

        '''
        logits_argmax = logits_argmax.cpu().numpy()
        dst = dst.cpu().numpy()
        bracket = bracket.cpu().numpy()
        for b in range(len(logits)):
            ind = bracket[b] == 1
            pred = logits_argmax[b][ind]
            gold = dst[b][ind]
            print(sentence_list[b])
            print(self.tokenizer.convert_ids_to_tokens(pred))
            print(self.tokenizer.convert_ids_to_tokens(gold))
            input()
        '''
        return acc_token.item(), acc_sent.item()
