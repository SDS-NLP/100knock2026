from knock80 import load_tokenizer
from knock81 import load_mlm, mask_probs

# 82. マスクのtop-k予測
# 81の確率の表(全語彙でsoftmax済み)から topk で上位10件を取る。
# 注: softmaxを上位10個だけに掛けると「10個の中で合計1」に水増しされる。
#     確率化は全語彙 → 選抜は後、の順。

TOP_K = 10


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_mlm()

    text = f"The movie was full of {tokenizer.mask_token}."
    probs = mask_probs(text, tokenizer, model)[0]

    # topk は値(確率)と添字(トークンID)を降順で同時に返す。
    top = probs.topk(TOP_K)
    for p, i in zip(top.values, top.indices):
        print(f"{tokenizer.convert_ids_to_tokens(i.item()):<12} {p:.4f}")
