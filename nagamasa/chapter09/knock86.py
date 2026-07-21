import torch

from knock80 import load_tokenizer
from knock85 import TRAIN_PATH, load_dataset

# 86. ミニバッチの作成
# 85のID列(長さ不揃い)をバッチ内最長に合わせ、後ろに [PAD] を足して揃える。
# attention_mask は実トークン=1 / PAD=0。モデル入力と labels は分けて持つ
# (lossは87で自分で書くため、HF流の「labelsも渡してloss任せ」にはしない)。


def make_batch(examples, pad_id):
    """{input_ids, attention_mask} と labels のテンソルを返す。87でも使う。"""
    max_len = max(len(d["input_ids"]) for d in examples)
    ids = [d["input_ids"] + [pad_id] * (max_len - len(d["input_ids"])) for d in examples]
    mask = [[1] * len(d["input_ids"]) + [0] * (max_len - len(d["input_ids"])) for d in examples]
    inputs = {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(mask),
    }
    labels = torch.tensor([d["label"] for d in examples])
    return inputs, labels


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    train = load_dataset(TRAIN_PATH, tokenizer)

    inputs, labels = make_batch(train[:4], tokenizer.pad_token_id)

    # ids と mask は別々に組んでいるので突き合わせる: PADの位置とmask=0の位置が一致するか。
    assert ((inputs["input_ids"] == tokenizer.pad_token_id) == (inputs["attention_mask"] == 0)).all()

    print("shape:", tuple(inputs["input_ids"].shape))  # (4, 15) = (バッチ, バッチ内最長)
    print("input_ids:", inputs["input_ids"], sep="\n")
    print("attention_mask:", inputs["attention_mask"], sep="\n")
    print("labels:", labels)  # tensor([0, 0, 1, 0])
