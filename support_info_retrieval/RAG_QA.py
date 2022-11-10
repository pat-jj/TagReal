from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset

import torch


# wiki_dataset = load_dataset("xiang's", "20220301.en")
# wiki_dataset = wiki_dataset['train']
# device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    # "facebook/rag-token-nq", index_name="custom", indexed_dataset=wiki_dataset
    "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
)
# initialize with RagRetriever to do everything in one forward call
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)


def ask_question(question):

    retriever_input_ids = model.retriever.question_encoder_tokenizer.batch_encode_plus(
        [question],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to(device)

    # Occasionally it isn't able to return 5 answers
    # In that case, keep decreasing the number until it succeeds
    num_return = 5
    while num_return > 0:
        try:
            generated = model.generate(retriever_input_ids, num_beams=5, num_return_sequences=num_return)
        except RuntimeError:
            num_return -= 1
        else:
            break
    answers = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return answers


def main():

    questions = [
        "Who is the president of the United States in 2018?",
        "How far away is the moon from Earth?",
        "Where do dolphins live?",
        "What do monkeys like to eat?",
        "In what stadium does Manchester United play?",
        "What is Microsoft?",
    ]

    for q in questions:
        results = ask_question(q)
        print(results)


if __name__ == "__main__":
        main()



