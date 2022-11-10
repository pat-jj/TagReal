import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
import torch.nn.functional as F

import re

from transformers import AutoTokenizer
from p_tuning.models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from p_tuning.prompt_encoder import *
from torch import distributed as dist



class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, device_ids, template, tokenizer_src, relation_num):
        super().__init__()
        self.args = args
        self.device = device
        self.relation_num = relation_num
        self.tokenizer = tokenizer_src
        self.template = template
        self.device_ids = device_ids

        self.model = create_model(self.args)
        
        ##modified
        print(torch.cuda.device_count())  # 打印gpu数量
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        print('world_size', torch.distributed.get_world_size()) # 打印当前进程数
        torch.cuda.set_device(self.args.local_rank)
        
        
        if len(self.device_ids) > 1:
            self.model = self.model.cuda(args.local_rank)
#             self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.device_ids)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=False, 
                                                     broadcast_buffers=False)
            
        self.model.module.to(self.device)
        self.model.to(self.device)
        
        
        
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
            
        ## modified
        self.embeddings = get_embedding_layer(self.args, self.model.module)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = KEPromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args, self.relation_num)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def embed_input(self, queries, rs_tensor):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder(rs_tensor)
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[bidx, i, :]
        return raw_embeds

    def forward_classification(self, texts, rs, labels, return_candidates=False):
        if self.args.model_name == 'luke':
            return self.forward_classification_luke(texts, rs, labels, return_candidates)
        bz = len(texts)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        queries = [torch.LongTensor(self.prompt_encoder.get_query(texts[i], rs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        # construct label ids
        attention_mask = queries != self.pad_token_id
        rs_tensor = torch.LongTensor(rs).to(self.device)

        # get embedded input
        inputs_embeds = self.embed_input(queries, rs_tensor)

        output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                            attention_mask=attention_mask.to(self.device).bool(),
                            labels=labels.to(self.device))
        loss, logits = output.loss, output.logits
        acc = torch.sum(torch.argmax(logits, dim=-1) == labels.to(self.device))

        return loss, float(acc) / bz, (labels.tolist(), torch.argmax(logits, dim=-1).tolist(), logits)
    
    def forward_classification_luke(self, texts, rs, labels, return_candidates=False):
        bz = len(texts)
        input_texts = []
        input_entities = []
        input_entity_spans = []

        for i in range(bz):
            text = texts[i]
            contents = text.split('\t\t')
            e1, e2 = contents[1], contents[3]
            sentence = ''.join(contents)
            input_texts.append(sentence)
            input_entities.append([e1, e2])
            input_entity_spans.append([(sentence.find(e1), sentence.find(e1) + len(e1)), (sentence.find(e2), sentence.find(e2) + len(e2))])
        
        encoding = self.tokenizer(
            input_texts, 
            entities=input_entities, 
            entity_spans=input_entity_spans, 
            add_prefix_space=True, 
            padding=True, 
            return_tensors="pt"
            )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        output = self.model(**encoding, labels=labels.to(self.device))

        loss, logits = output.loss, output.logits
        acc = torch.sum(torch.argmax(logits, dim=-1) == labels.to(self.device))

        return loss, float(acc) / bz, (labels.tolist(), torch.argmax(logits, dim=-1).tolist(), logits)