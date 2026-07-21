import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 新しいセッションの場合はDriveから復元 ---
# from google.colab import drive
# drive.mount('/content/drive')
# path = "/content/drive/MyDrive/nlp100/knock87_best"
# tokenizer = AutoTokenizer.from_pretrained(path)
# model = AutoModelForSequenceClassification.from_pretrained(path).to(device)

model.eval()

sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits              # (5, 2)

probs = torch.softmax(logits, dim=-1)            # 各文の [ネガ確率, ポジ確率]
preds = logits.argmax(dim=-1)                    # 0=negative, 1=positive

for sent, pred, prob in zip(sentences, preds, probs):
    label = "positive" if pred.item() == 1 else "negative"
    print(f"{label:8s} (neg {prob[0].item():.4f} / pos {prob[1].item():.4f})  {sent}")