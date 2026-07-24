#General Language Understanding Evaluation (GLUE) ベンチマークで配布されているStanford Sentiment Treebank (SST) から訓練セット（train.tsv）と開発セット（dev.tsv）のテキストと極性ラベルと読み込み、さらに全てのテキストはトークン列に変換せよ。

from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_df = pd.read_csv("SST-2/train.tsv", sep = "\t")
dev_df = pd.read_csv("SST-2/dev.tsv", sep = "\t")

train = []
dev = []

for sentence, label in zip(train_df["sentence"], train_df["label"]):
    
    tokens = tokenizer.tokenize(sentence)
    
    train.append({
        "sentence": sentence,
        "label": int(label),
        "tokens": tokens
    })

for sentence, label in zip(dev_df["sentence"], dev_df["label"]):
    
    tokens = tokenizer.tokenize(sentence)
    
    dev.append({
        "sentence": sentence,
        "label": int(label),
        "tokens": tokens
    })

if __name__ == "__main__":
    
    print("train_size:", len(train))
    print("dev_size:", len(dev))
    
    print(train[0])
    print(dev[0])