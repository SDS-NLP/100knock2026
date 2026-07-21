import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    text = "The movie was full of [MASK]."
    inputs = tokenizer(text, return_tensors="pt")   # PyTorchテンソルで返す
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits                    # (1, 系列長, 30522)

    # [MASK] の位置を探す
    mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # その位置のロジットが最大のトークン = 最有力候補
    pred_id = logits[0, mask_pos].argmax(dim=-1)
    print(tokenizer.decode(pred_id))

if __name__ == "__main__":
    main()