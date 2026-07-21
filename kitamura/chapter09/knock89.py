import torch.nn as nn
from transformers import AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
from tqdm import tqdm

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
    padding=True,          
    truncation=True,       
    return_tensors="pt"    
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
        padding=True,          
        truncation=True,       
        return_tensors="pt"   
    )
dev_labels_tensor = torch.tensor(dev_labels)

batch_size = 8
train_dataset = TensorDataset(train_encoded["input_ids"], train_encoded["attention_mask"])
dev_dataset = TensorDataset(dev_encoded["input_ids"], dev_encoded["attention_mask"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 2

class MaxPoolBertClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2):
        super(MaxPoolBertClassifier, self).__init__()
        # BERTをロード
        self.bert = AutoModel.from_pretrained(model_name)
        # 抽出した7ベクトルを2クラスに分ける線形層
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_stateの形状
        hidden_state = outputs.last_hidden_state 
        
        # マスクが0の部分には非常に小さな値（-1e9）を入れる
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        hidden_state = hidden_state.masked_fill(mask_expanded == 0, -1e9)
        
        # dim=1の方向で最大値を取得する
        max_pooled = torch.max(hidden_state, dim=1)[0]
        
        # 予測スコア
        logits = self.classifier(max_pooled)
        
        # labelsが渡された場合は、CrossEntropyLossも計算して返す
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        return loss, logits


model_89 = MaxPoolBertClassifier()
optimizer_89 = AdamW(model_89.parameters(), lr=2e-5)

#  学習
print("【Task 89: 最大値プーリング(Max Pooling)を用いた学習を開始】")

for epoch in range(epochs):
    # --- 学習モード ---
    model_89.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        b_input_ids, b_mask, b_labels = [b for b in batch]
        
        optimizer_89.zero_grad()
        # カスタムモデルのforwardが呼ばれる
        loss, _ = model_89(input_ids=b_input_ids, attention_mask=b_mask, labels=b_labels)
        loss.backward()
        optimizer_89.step()
        
   
    model_89.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dev_loader:
            b_input_ids, b_mask, b_labels = [b.to(device) for b in batch]
            _, logits = model_89(input_ids=b_input_ids, attention_mask=b_mask)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)
            
    print(f"-> Task 89 | Epoch {epoch+1} | Dev Accuracy (正解率): {correct / total:.4f}\n")