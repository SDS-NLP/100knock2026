import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knock87 import DEVICE, SAVE_DIR

# 88. 極性分析
# 87で保存したモデルを読み戻して5文の極性を予測する(先に knock87.py の実行が必要)。
# from_pretrained はHub名でもローカルのディレクトリでも同じ顔で動く。
# softmaxは順位を変えないが、迷いの度合いを読むために確率も添える。
#
# 結果: incomprehensibilities → negative 0.996(サブワード6片からでも極性を読めた)。
#       fun/excitement → positive 0.999、crap/rubbish → negative 0.997。
# 参考: ファインチューニング前の bert-base-uncased だと5文全部 negative p≈0.7
#       (ヘッドがランダム初期化のままなので、出力は極性と無相関)。

SENTENCES = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]
LABELS = ["negative", "positive"]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
    model.eval()

    inputs = tokenizer(SENTENCES, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1)

    for text, p in zip(SENTENCES, probs):
        print(f"{text:<50} {LABELS[p.argmax()]:<10} (p={p.max():.3f})")
