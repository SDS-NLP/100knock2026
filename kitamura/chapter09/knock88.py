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
sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        b_input_ids, b_mask, b_labels = [b for b in batch]
        outputs = model(input_ids=b_input_ids, attention_mask=b_mask)

        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_mask, labels=b_labels)
        loss = outputs.loss
        totsl_loss += loss.item()

        loss.backward()
        optimizer.step()


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)

            pred =torch.argmax(outputs.logits, dim=1).item()
            label_str = "Positive" if pred == 1 else "Negative"
            print(f"[{label_str}] {sentence}")
            
