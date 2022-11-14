from functools import partial

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, DPRContextEncoder, \
    DPRContextEncoderTokenizer
from datasets import load_dataset, load_from_disk, Features, Value, Sequence
import faiss

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


def build_corpus_dataset(load_available_dataset=False):

    if load_available_dataset:
        dataset = load_dataset('pat-jj/nyt10_corpus')['train']

    else:
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

        try:
            dataset.push_to_hub('pat-jj/nyt10_corpus')
        except:
            print('Fail to push the dataset to model hub')

    faiss_num_dim = 768
    faiss_num_links = 128
    index = faiss.IndexHNSWFlat(faiss_num_dim, faiss_num_links, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)

    return dataset


def load_retriever_and_generator(dataset, load_model=False):
    rag_model_name = "facebook/rag-token-nq"

    if load_model:
        retriever = RagRetriever.from_pretrained("pat-jj/nyt10-finetuned-rag-retriever", index_name="exact")
        model = RagTokenForGeneration.from_pretrained("pat-jj/nyt10-finetuned-rag-generator", index_name="exact")
    else:
        retriever = RagRetriever.from_pretrained(
            rag_model_name, index_name="custom", indexed_dataset=dataset
        )
        # initialize with RagRetriever to do everything in one forward call
        model = RagTokenForGeneration.from_pretrained(rag_model_name, retriever=retriever).to(device)

        # try:
        #     retriever.push_to_hub("pat-jj/nyt10-finetuned-rag-retriever")
        #     print("Successfully push the retriever to the model hub!")
        #     model.push_to_hub("pat-jj/nyt10-finetuned-rag-generator")
        #     print("Successfully push the generator to the model hub!")
        # except:
        #     print("fail to push the retriever/generator to model hub")

    tokenizer = RagTokenizer.from_pretrained(rag_model_name)

    return model, tokenizer, retriever


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

def get_prompt_for_relation_qa(head, tail, relation):

    prompt = None
    prompts = {
        '/people/person/nationality': "Nationality of [placeholder]?",
        '/location/location/contains': "Where is [placeholder] located?",
        # '/people/person/place_lived',
        # '/people/deceased_person/place_of_death',
        # '/people/person/ethnicity',
        # '/people/ethnicity/people',
        # '/business/person/company',
        # '/people/person/religion',
        # '/location/neighborhood/neighborhood_of',
        # '/business/company/founders',
        # '/people/person/children',
        # '/location/administrative_division/country',
        # '/location/country/administrative_divisions',
        # '/business/company/place_founded',
        # '/location/us_county/county_seat'
    }
    prompt = prompts[relation]

    return prompt


def build_question_for_triple(triple):
    head, relation, tail = triple[0], triple[1], triple[2]
    question = get_prompt_for_relation_qa(head, tail, relation)
    return question


def main():
    dataset = build_corpus_dataset(load_available_dataset=True)
    model, tokenizer, retriever = load_retriever_and_generator(dataset=dataset, load_model=False)

    questions = [
        "When was Google founded?",
        "Where is Hunan located?",
        "Who founded Microsoft?",
        "Who founded Netflix?"
    ]

    for q in questions:
        results = ask_question(model, tokenizer, q)
        print(results)


if __name__ == "__main__":
    main()
