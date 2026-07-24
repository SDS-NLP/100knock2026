#"The movie was full of [MASK]."の"[MASK]"に埋めるのに適切なトークン上位10個と、その確率（尤度）を求めよ。

import torch
import knock80
import knock81

tokenizer = knock80.tokenizer
model = knock81.model

text = "The movie was full of [MASK]."

mask_token_logits = knock81.mask_token_logits

mask_token_prob = torch.softmax(mask_token_logits, dim = 1)

top10_tokens = torch.topk(mask_token_prob, 10, dim = 1).indices[0].tolist() #probの上位10単語を取り出す(mask_token_logitsは[1(MASK数),30522(語彙)]なので、dim=1で語彙の方向に対してトークンIDをpythonのリストの形で取り出す,MASKは1つなのでindices[0])

if __name__ == "__main__":
    
    for token_id in top10_tokens:
        
        token = tokenizer.decode([token_id]) #トークナイザーを用いてトークンIDを文字列に変換(decodeはリストを受け取る)
        prob = mask_token_prob[0, token_id].item()
        
        print(token, prob)
        