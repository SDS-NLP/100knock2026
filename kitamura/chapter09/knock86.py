from transformers import AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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


minibatch_inputs = {
    'input_ids': train_encoded['input_ids'][:4],
    'attention_mask': train_encoded['attention_mask'][:4]
}

minibatch_labels = train_labels_tensor[:4]

print("input_ids :", minibatch_inputs['input_ids'].shape)
print("labels :", minibatch_labels.shape)