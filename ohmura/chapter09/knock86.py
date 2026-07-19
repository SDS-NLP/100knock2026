import pandas as pd
from transformers import BertTokenizer

train_path = '../chapter07/SST-2/train.tsv'
df = pd.read_csv(train_path, sep='\t').head(4)
sentences = df['sentence'].tolist()
labels = df['label'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

batch = tokenizer(
    sentences,
    padding=True,
    return_tensors='pt'
)

print("=== 元のテキスト ===")
for i, text in enumerate(sentences):
    print(f"[{i}] {text}")

print("\n=== トークンID (input_ids) ===")
print(batch['input_ids'])

print("\n=== アテンションマスク (attention_mask) ===")
print(batch['attention_mask'])