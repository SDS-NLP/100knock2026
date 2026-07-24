#問題87でファインチューニングされたモデルを用いて、以下の文の極性を予測せよ。
#"The movie was full of incomprehensibilities."
#"The movie was full of fun."
#"The movie was full of excitement."
#"The movie was full of crap."
#"The movie was full of rubbish."

from transformers import BertTokenizer, BertForSequenceClassification
import torch

sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

tokenizer = BertTokenizer.from_pretrained("knock_bert")
model = BertForSequenceClassification.from_pretrained("knock_bert")

model.eval()

inputs = tokenizer(
    sentences,
    padding = True,
    truncation = True,
    return_tensors = "pt"
)

with torch.no_grad():
    
    outputs = model(**inputs) #各文がpositive/negativeどちらっぽいかのスコア([文の数,ラベル数])を返す

preds = torch.argmax(outputs.logits, dim = 1) #値が大きい方のインデックスを返す(outputs.logitsは[negativeスコア,positiveスコア]になっているので、negatuveが大きいと0,positiveが大きいと1)
probs = torch.softmax(outputs.logits, dim = 1) #logitsを確率に変換

for sentence, pred, prob in zip(sentences, preds, probs):
    
    pred_id = pred.item()
    label = "positive" if pred_id == 1 else "negative" #pred_idが1のときpositive,そうでないときnegative
    confidence = prob[pred_id].item() #予測したポシネガの確率
    
    print(sentence, label, confidence)