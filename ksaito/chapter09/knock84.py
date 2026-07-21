import torch
from transformers import AutoModel

from knock80 import MODEL_NAME, load_tokenizer
from knock83 import SENTENCES, print_pairwise_cosine_similarities


def mean_vectors(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")
    model.eval()
    with torch.inference_mode():
        hidden_states = model(**inputs).last_hidden_state

    # [PAD]は平均から除外する。[CLS]と[SEP]はBERTのトークンとして含める。
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)


def main():
    tokenizer = load_tokenizer()
    model = AutoModel.from_pretrained(MODEL_NAME)
    vectors = mean_vectors(SENTENCES, tokenizer, model)

    for i, sentence in enumerate(SENTENCES, start=1):
        print(f"{i}: {sentence}")
    print_pairwise_cosine_similarities(SENTENCES, vectors)


if __name__ == "__main__":
    main()
