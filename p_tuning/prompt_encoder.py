import torch
import torch.nn as nn

class KEPromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args, relation_num):
        super().__init__()
        self.device = device
        self.template = template
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        self.relation_num = relation_num
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.spell_length
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices_relation = torch.LongTensor(list(range(sum(self.template)))).to(self.device)
        # embedding
        self.embedding_relation = torch.nn.Embedding(sum(self.template) * self.relation_num, self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self, rs_tensor):
        if sum(self.template) == 0:
            return None

        # bz x template
        seq_indices_relation_spec = self.seq_indices_relation.unsqueeze(0) + rs_tensor.unsqueeze(-1) * sum(self.template)
        # bz x template x dim
        input_embeds = self.embedding_relation(seq_indices_relation_spec)
        
        # output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        # return output_embeds

        return input_embeds

    def get_query(self, texts, rs, prompt_tokens):
        contents = texts.split('\t\t')
        ans_list = [self.tokenizer.cls_token_id]

        # length = 5 for triple setting
        for i in range(len(contents)):
            ans_list += prompt_tokens * self.template[i]
            if len(contents[i]) != 0:
                if len(ans_list) == 1:
                    ans_list += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(contents[i]))
                else:
                    ans_list += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + contents[i]))
        ans_list += prompt_tokens * self.template[-1]
        ans_list += [self.tokenizer.sep_token_id]

        return [ans_list]