import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from itertools import combinations

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    sentences = [
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]

    # 4文まとめてトークン化(長さはpaddingで揃える)
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state        # (4, seq_len, 768)
    cls_vecs = last_hidden[:, 0, :]                # (4, 768)  各文の[CLS]は必ず0番目

    for i, j in combinations(range(len(sentences)), 2):
        sim = F.cosine_similarity(cls_vecs[i], cls_vecs[j], dim=0).item()
        print(f"{sim:.4f}  |  {sentences[i]}  vs  {sentences[j]}")

if __name__ == "__main__":
    main()