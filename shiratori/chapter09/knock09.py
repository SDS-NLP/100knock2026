import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BertModel,
    TrainingArguments,
    Trainer,
)
import evaluate


class BertMeanPoolingClassifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()

        # 学習済みBERTを読み込む
        self.bert = BertModel.from_pretrained(checkpoint)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)
        hidden = hidden * mask

        pooled = hidden.sum(dim=1) / mask.sum(dim=1)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def main():

    checkpoint = "./results/checkpoint-16838"

    train = pd.read_csv("data/SST-2/train.tsv", sep="\t")
    dev = pd.read_csv("data/SST-2/dev.tsv", sep="\t")

    train_dataset = Dataset.from_pandas(train)
    dev_dataset = Dataset.from_pandas(dev)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    model = BertMeanPoolingClassifier(checkpoint)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.to(device)

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
        per_device_eval_batch_size=16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    result = trainer.evaluate()

    print("Validation Accuracy:", result["eval_accuracy"])

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

    predictions = torch.argmax(outputs["logits"], dim=1)

    labels = ["Negative", "Positive"]

    print("\nPrediction Results")
    print("-" * 40)

    for sentence, pred in zip(sentences, predictions):
        print(f"Sentence : {sentence}")
        print(f"Prediction: {labels[pred.item()]}")
        print()


if __name__ == "__main__":
    main()


# Validation Accuracy: 0.4908256880733945

# Prediction Results
# ----------------------------------------
# Sentence : The movie was full of incomprehensibilities.
# Prediction: Negative

# Sentence : The movie was full of fun.
# Prediction: Negative

# Sentence : The movie was full of excitement.
# Prediction: Negative

# Sentence : The movie was full of crap.
# Prediction: Negative

# Sentence : The movie was full of rubbish.
# Prediction: Negative
