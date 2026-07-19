import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate


def main():

    train = pd.read_csv("data/SST-2/train.tsv", sep="\t")
    dev = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

    train_dataset = Dataset.from_pandas(train)
    dev_dataset = Dataset.from_pandas(dev)

    checkpoint = "./results/checkpoint-16838"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    dev_dataset = dev_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    dev_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=1)
        return accuracy.compute(
            predictions=predictions,
            references=labels,
        )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    result = trainer.evaluate()
    print(f"Validation Accuracy: {result['eval_accuracy']:.4f}")

    # -------------------------
    # 問88：極性分析
    # -------------------------
    sentences = [
        "The movie was full of incomprehensibilities.",
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=1)

    labels = ["Negative", "Positive"]

    print("\nPrediction Results")
    print("-" * 40)

    for sentence, pred in zip(sentences, predictions):
        print(f"Sentence : {sentence}")
        print(f"Prediction: {labels[pred.item()]}")
        print()


if __name__ == "__main__":
    main()

# oading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 4353.07it/s]
# Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 67349/67349 [00:03<00:00, 21829.27 examples/s]
# Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 872/872 [00:00<00:00, 10442.50 examples/s]
# /Users/shiratorihanae/Downloads/2026/2026春夏/100本ノック/100knock2026/shiratori/venv/lib/python3.11/site-packages/torch/utils/data/dataloader.py:752: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
#   super().__init__(loader)
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 218/218 [00:28<00:00,  7.55it/s]
# Validation Accuracy: 0.9255

# Prediction Results
# ----------------------------------------
# Sentence : The movie was full of incomprehensibilities.
# Prediction: Negative

# Sentence : The movie was full of fun.
# Prediction: Positive

# Sentence : The movie was full of excitement.
# Prediction: Positive

# Sentence : The movie was full of crap.
# Prediction: Negative

# Sentence : The movie was full of rubbish.
# Prediction: Negative
