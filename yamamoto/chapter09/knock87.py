#訓練セットを用い、事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。検証セット上でファインチューニングされたモデルの正解率を計測せよ。

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.optim import AdamW
from tqdm import tqdm
import knock85

def collate(data):
    
    sentences = [example["sentence"] for example in data]
    labels = [example["label"] for example in data]

    batch = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    batch["labels"] = torch.tensor(labels, dtype=torch.long)

    return batch

train = knock85.train[:1000]
dev = knock85.dev[:200]

train_batch = torch.utils.data.DataLoader(
    dataset = train,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate
)

dev_batch = torch.utils.data.DataLoader(
    dataset = dev,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate
)

model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name)

model = BertForSequenceClassification.from_pretrained( #分類用BERT
    model_name,
    num_labels = 2
)

optimizer = AdamW(model.parameters(), lr = 2e-5)

model.train()

for epoch in range(3):
    
    total_loss = 0
    
    for batch in tqdm(train_batch):
        
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train)
    print(epoch, "平均損失：", avg_loss)

model.save_pretrained("knock_bert")
tokenizer.save_pretrained("knock_bert")

model.eval()

score = 0

with torch.no_grad():
    
    for batch in dev_batch:
        
       labels = batch["labels"]
       
       outputs = model(**batch)
       
       preds = torch.argmax(outputs.logits, dim = 1)
       
       score += (preds == labels).sum().item()
    
    accuracy = score / len(dev)

print("accracy", accuracy)    