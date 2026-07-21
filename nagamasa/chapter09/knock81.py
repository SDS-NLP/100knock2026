import torch
from transformers import AutoModelForMaskedLM

from knock80 import MODEL_NAME, load_tokenizer

# 81. マスクの予測
# "The movie was full of [MASK]." を埋める最適トークンを求める。
# logits (1, 系列長, 語彙数) から [MASK] 位置の行を取り出し、softmax → argmax。
# マスク記号はモデル固有なので tokenizer.mask_token から取る。


def load_mlm(model_name=MODEL_NAME):
    """MLMヘッド付きBERT。eval() で dropout を止める。"""
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    return model


def mask_probs(text, tokenizer, model):
    """各 [MASK] 位置の語彙確率分布 (マスク数, 語彙数) を返す。82でも使う。"""
    inputs = tokenizer(text, return_tensors="pt")

    # 推論のみ。勾配の記録は不要。
    with torch.no_grad():
        outputs = model(**inputs)

    # [MASK] の位置一覧。nonzero() の (件数, 1) を squeeze で (件数,) に潰す。
    positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().squeeze(-1)

    # 位置テンソルを添字にすると各マスク位置の行がまとめて取れる(マスク数, 語彙数)。
    return outputs.logits[0, positions].softmax(dim=-1)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_mlm()

    text = f"The movie was full of {tokenizer.mask_token}."
    probs = mask_probs(text, tokenizer, model)[0]  # マスクは1個なので先頭の分布だけ

    best_id = probs.argmax().item()
    print(tokenizer.convert_ids_to_tokens(best_id), f"(p={probs[best_id]:.3f})")
