"""
87. ファインチューニング
訓練セットを用い、事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。検証セット上でファインチューニングされたモデルの正解率を計測せよ。
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
import evaluate


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_path = "../chapter08/SST-2/train.tsv"
    dev_path = "../chapter08/SST-2/dev.tsv"

    train_df = pd.read_csv(train_path, sep="\t", header=0)
    dev_df = pd.read_csv(dev_path, sep="\t", header=0)

    # データセットの作成
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(dev_df)

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"最終評価結果: {eval_results}")


if __name__ == "__main__":
    main()