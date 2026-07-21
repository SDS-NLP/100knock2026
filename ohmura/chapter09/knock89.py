import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from tqdm import tqdm

class SST2Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CustomBertClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
        hidden_state = hidden_state.masked_fill(mask_expanded == 0, -1e9)
        
        max_pooled, _ = torch.max(hidden_state, dim=1)
        logits = self.classifier(max_pooled)
        
        return logits

def main():
    train_df = pd.read_csv('../chapter07/SST-2/train.tsv', sep='\t')
    dev_df = pd.read_csv('../chapter07/SST-2/dev.tsv', sep='\t')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = CustomBertClassifier()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    train_dataset = SST2Dataset(train_df['sentence'].tolist(), train_df['label'].tolist(), tokenizer)
    dev_dataset = SST2Dataset(dev_df['sentence'].tolist(), dev_df['label'].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 1
    
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"開発セットの正解率: {accuracy:.4f}")

if __name__ == '__main__':
    main()