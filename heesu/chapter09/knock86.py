from transformers import AutoTokenizer

from knock85 import DATA_DIR, MODEL_NAME, load_split

N = 4  # number of leading training examples to batch together


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts, labels = load_split(DATA_DIR / "train.tsv")
    batch_texts = texts[:N]
    batch_labels = labels[:N]

    # Token lengths differ from sentence to sentence before padding.
    raw_lengths = [len(tokenizer(t)["input_ids"]) for t in batch_texts]

    # Pad the batch so every sequence has the same length (the longest in the
    # batch), returning PyTorch tensors ready to feed to the model.
    batch = tokenizer(batch_texts, padding=True, return_tensors="pt")

    for i, text in enumerate(batch_texts):
        print(f"[{i}] label={batch_labels[i]} raw_len={raw_lengths[i]} | {text}")

    print("\ninput_ids shape     :", tuple(batch["input_ids"].shape))
    print("attention_mask shape:", tuple(batch["attention_mask"].shape))
    print("\ninput_ids (0 = [PAD]):")
    print(batch["input_ids"])
    print("\nattention_mask (1 = real token, 0 = padding):")
    print(batch["attention_mask"])


if __name__ == "__main__":
    main()
