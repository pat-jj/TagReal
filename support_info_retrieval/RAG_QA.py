from functools import partial

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, DPRContextEncoder, \
    DPRContextEncoderTokenizer
from datasets import load_dataset, load_from_disk, Features, Value, Sequence

import torch
import re
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


def split_into_sentences(text, n_sents=5):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    if len(sentences[-1]) == 0:
        sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    final_sentences = []
    for i in range(0, len(sentences), n_sents):
        final_sentences.append(" ".join(sentences[i:i + n_sents]))
    return final_sentences


def make_chunked_file(chunked_filename, corpus_filename, split_func):
    """
  of a new chunked json file to be loaded into the dataset object.
  """
    with open(corpus_filename) as corpus_file:
        with open(chunked_filename, "w") as chunked_file:
            for jline in json.load(corpus_file):
                for passage in split_func(jline["sentence"]):
                    json.dump({
                        "text": passage,
                        "title": jline["relation"],
                    }, chunked_file)
                    chunked_file.write("\n")


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizer) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def add_len(example):
    example["len"] = len(example["title"] + example["text"])
    return example


def build_corpus_dataset():
    f = open('../Volumes/Aux/Downloaded/Data-Upload/FB60K+NYT10/text/train.json')
    data_corpus = json.load(f)

    f = open('../Volumes/Aux/Downloaded/Data-Upload/FB60K+NYT10/text/test.json')
    tmp = json.load(f)

    data_corpus = data_corpus + tmp

    corpus_filename = 'nyt10_corpus.json'
    chunked_filename = 'nyt10_corpus_chunked.json'
    with open(corpus_filename, 'w') as dump_file:
        json.dump(data_corpus, dump_file)

    make_chunked_file(
        chunked_filename=chunked_filename,
        corpus_filename=corpus_filename,
        split_func=split_into_sentences
    )

    dataset = load_dataset(
        "json",
        data_files=chunked_filename,
        split="train",
    )
    dataset = dataset.map(add_len).sort("len").remove_columns(["len"])

    torch.set_grad_enabled(False)

    dpr_model_name = "facebook/dpr-ctx_encoder-multiset-base"
    batch_size = 32

    ctx_encoder = DPRContextEncoder.from_pretrained(dpr_model_name).to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_model_name)
    new_features = Features({
        "text": Value("string"),
        "title": Value("string"),
        "embeddings": Sequence(Value("float32")),
    })

    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=batch_size,
        features=new_features,
    )

    return dataset


def load_retriever_and_generator(dataset):
    rag_model_name = "facebook/rag-token-nq"
    tokenizer = RagTokenizer.from_pretrained(rag_model_name)
    retriever = RagRetriever.from_pretrained(
        rag_model_name, index_name="custom", indexed_dataset=dataset
    )
    # initialize with RagRetriever to do everything in one forward call
    model = RagTokenForGeneration.from_pretrained(rag_model_name, retriever=retriever).to(device)

    return model, tokenizer


def ask_question(model, tokenizer, question):
    retriever_input_ids = model.retriever.question_encoder_tokenizer.batch_encode_plus(
        [question],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to(device)

    question_enc_outputs = model.rag.question_encoder(retriever_input_ids)
    question_enc_pool_output = question_enc_outputs[0]

    result = model.retriever(
        retriever_input_ids,
        question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
        prefix=model.rag.generator.config.prefix,
        n_docs=model.config.n_docs,
        return_tensors="pt",
    )

    all_docs = model.retriever.index.get_doc_dicts(result.doc_ids)
    titles = []
    for docs in all_docs:
        titles.extend([title for title in docs["title"]])

    # Occasionally it isn't able to return 5 answers
    # In that case, keep decreasing the number until it succeeds
    num_return = 5
    generated = None

    while num_return > 0:
        try:
            generated = model.generate(retriever_input_ids, num_beams=5, num_return_sequences=num_return)
        except RuntimeError:
            num_return -= 1
        else:
            break
    answers = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return {
        'answers': answers,
        'titles': titles
    }


def main():
    dataset = build_corpus_dataset()
    model, tokenizer = load_retriever_and_generator(dataset=dataset)

    questions = [
        "Who is the president of the United States in 2018?",
        "How far away is the moon from Earth?",
        "Where do dolphins live?",
        "What do monkeys like to eat?",
        "In what stadium does Manchester United play?",
        "What is Microsoft?",
    ]

    for q in questions:
        results = ask_question(model, tokenizer, q)
        print(results)


if __name__ == "__main__":
    main()
