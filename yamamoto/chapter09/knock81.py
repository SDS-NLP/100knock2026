#"The movie was full of [MASK]."の"[MASK]"を埋めるのに最も適切なトークンを求めよ。

from transformers import BertForMaskedLM
import torch
import knock80

model_name = knock80.model_name
tokenizer = knock80.tokenizer

model = BertForMaskedLM.from_pretrained(model_name)

text = "The movie was full of [MASK]."

inputs = tokenizer(text, return_tensors = "pt") #入力としてtextをBERTモデルでトークン化したものを準備(返り値はpytorchテンソル)

with torch.no_grad():
    
    outputs = model(**inputs) #BERTモデル(今回はトークナイザーと同じ)により、各トークン位置においてどの単語が入りそうかを予測(outputsには各位置についてモデルの語彙の数だけスコアが格納される)

logits = outputs.logits #outputsのうちスコアの部分

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1] #MASKのトークンIDがあるインデックス(torch.whereの返り値は[batch,token位置]の2次元)

mask_token_logits = logits[0, mask_token_index, :] #MASKの位置における全ての語彙のスコア

best_token_id = torch.argmax(mask_token_logits, dim = 1).item()

best_token = tokenizer.decode([best_token_id])

if __name__ == "__main__":
    
    print(best_token)