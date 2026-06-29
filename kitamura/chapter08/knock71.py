import os
import urllib.request
import zipfile
import torch

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

train = []
with open("./SST-2/train.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        train.append(line.strip().split("\t"))

dev = []
with open("./SST-2/dev.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        dev.append(line.strip().split("\t"))

file_name = "word2vec_vocab_embedding.pt"
loaded_data = torch.load(file_name)  # 環境によっては引数に weights_only=False が必要です
word2id = loaded_data['word2id']


def create_SST_dataset(SST_data):
    dataset = []
    skip = 0

    for row in SST_data:
        text = row[0]
        label = row[1]

        tokens = text.split()
        input_ids = []
        for token in tokens:
            if token in word2id:
                input_ids.append(word2id.get(token))

        if len(input_ids) == 0:
            skip += 1
            continue

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = torch.tensor([float(label)], dtype=torch.float32)

        instance = {
            "text": text,
            "label": label,
            "input_ids": input_ids
        }

        dataset.append(instance)

    print(f"データ数：{len(dataset)}（skip : {skip}）")
    return dataset


train_dataset = create_SST_dataset(train)
dev_dataset = create_SST_dataset(dev)

print(train_dataset[0])

dataset_to_save = {
    "train": train_dataset,
    "dev": dev_dataset
}

file_name = "sst_datasets.pt"
torch.save(dataset_to_save, file_name)
print("save")