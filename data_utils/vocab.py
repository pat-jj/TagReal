import json
from os.path import join

def token_wrapper(args, token):
    if 'roberta' in args.model_name or 'gpt' in args.model_name or 'megatron' in args.model_name:
        return 'Ġ' + token
    else:
        return token

def get_vocab_by_strategy(args, tokenizer):
    return tokenizer.get_vocab()
