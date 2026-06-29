import os
import urllib.request
import zipfile
from transformers import AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
zip_filename = "SST-2.zip"
extract_dir = "./" 
target_file = "./SST-2/train.tsv" 

if not os.path.exists(target_file):
    urllib.request.urlretrieve(url, zip_filename)
    print("ダウンロード完了")

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("解凍完了。")

    if os.path.exists(zip_filename):
        os.remove(zip_filename)
  

    print("SSTダウンロード終了")

else:
    print("すでにダウンロードされています")

train_sentences = []
train_labels = []
with open("./SST-2/train.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        train_sentences.append(parts[0])
        train_labels.append(int(parts[1]))



train_encoded = tokenizer(
    train_sentences,
    padding=True,          # バッチ内の最大長に合わせて0埋め(PAD)する
    truncation=True,       # モデルの最大入力長を超えた場合は切り捨てる
    return_tensors="pt"    # PyTorchのテンソル形式で出力
)

train_labels_tensor = torch.tensor(train_labels)

dev_sentences = []
dev_labels = []
with open("./SST-2/dev.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            dev_sentences.append(parts[0])
            dev_labels.append(int(parts[1]))


dev_encoded = tokenizer(
        dev_sentences,
        padding=True,          # バッチ内の最大長に合わせて0埋め(PAD)する
        truncation=True,       # モデルの最大入力長を超えた場合は切り捨てる
        return_tensors="pt"    # PyTorchのテンソル形式で出力
    )
dev_labels_tensor = torch.tensor(dev_labels)
