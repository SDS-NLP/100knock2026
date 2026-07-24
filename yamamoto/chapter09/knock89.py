#問題87とは異なるアーキテクチャ（例えば[CLS]トークンを用いるか、各トークンの最大値プーリングを用いるなど）の分類モデルを設計し、事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。検証セット上でファインチューニングされたモデルの正解率を計測せよ。

#BertForSequenceClassificationを使わず、普通のBertModel+自作の線形分類器で分類

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import knock85

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)

train = knock85.train[:1000]
dev = knock85.dev[:200]

def collate(data):
    
    sentences = [example["sentence"] for example in data]
    labels = [example["label"] for example in data]

    batch = tokenizer(
        sentences,
        padding = True,
        truncation = True,
        max_length = 128,
        return_tensors = "pt"
    )

    batch["labels"] = torch.tensor(labels, dtype = torch.long)

    return batch

class BertMaxPoolClassifier(nn.Module):
    
    def __init__(self, model_name, num_labels = 2): #使用モデル,分類ラベルは2
        
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name) #分類用ではなく普通のBertModel(出力は主にトークン埋め込み)
        hidden_size = self.bert.config.hidden_size #隠れ層のベクトルの次元数(bert-base-uncasedは768次元)

        self.dropout = nn.Dropout(0.1) #学習時に一部の値をランダムに0にして過学習を抑制
        self.classifier = nn.Linear(hidden_size, num_labels) #hidden_size次元のベクトルをnum_labelsの数のクラスのスコアに変換
        self.loss_fn = nn.CrossEntropyLoss() #損失関数

    def forward(self, input_ids, attention_mask, token_type_ids = None, labels = None):
        
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask, #本文のトークンの位置は1,パディングの位置は0
            token_type_ids = token_type_ids
        )

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1) #attention_maskを[batch_size, sequence_length, 1]の形にする

        token_embeddings = token_embeddings.masked_fill(mask == 0, -1e9) #[PAD]の位置のベクトルを小さい値に置き換える(max poolingするから)

        pooled = torch.max(token_embeddings, dim = 1).values #全トークンのベクトルから次元ごとに最大値をとる

        logits = self.classifier(self.dropout(pooled)) #過学習を抑えながらクラス分類のスコアに変換

        if labels is not None: #正解ラベルがあるとき
            
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits} #予測だけのとき
    
    
train_loader = DataLoader(
    train,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate
)

dev_loader = DataLoader(
    dev,
    batch_size = 32,
    shuffle = False,
    collate_fn = collate
)

model = BertMaxPoolClassifier(model_name, num_labels = 2)

optimizer = AdamW(model.parameters(), lr = 2e-5)

epochs = 3

for epoch in range(epochs):
    
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        
        outputs = model(**batch)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch + 1}: train_loss = {avg_loss:.4f}")

model.eval()

correct = 0
total = 0

with torch.no_grad():
    
    for batch in dev_loader:
        
        labels = batch["labels"]

        outputs = model(**batch) #辞書型のキー名をそのまま引数名とし、valueを入力する
        logits = outputs["logits"]

        preds = torch.argmax(logits, dim = 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print(f"dev accuracy: {accuracy:.4f}")