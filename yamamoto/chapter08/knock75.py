#複数の事例が与えられたとき、これらをまとめて一つのテンソル・オブジェクトで表現する関数collateを実装せよ。与えられた複数の事例のトークン列の長さが異なるときは、トークン列の長さが最も長いものに揃え、0番のトークンIDでパディングをせよ。さらに、トークン列の長さが長いものから順に、事例を並び替えよ。

import torch
import knock71

def collate(data):
    
    padded = {} #パディングしたデータの辞書
    
    input_ids = []
    label = []
    
    sorted_data = sorted(data, key = lambda x: len(x["input_ids"]), reverse = True) #dataをIDトークン列の長い順にソート
    
    max_len = len(sorted_data[0]["input_ids"]) #最長のトークン列の長さを保存
    
    for example in sorted_data:
        
        pad = torch.tensor([0] * (max_len - len(example["input_ids"])), dtype = torch.long) #最長の長さに合わせてパディング要素(0)を作成(torch.long:整数テンソル型)
        
        input_ids.append(torch.cat([example["input_ids"], pad])) #パディング要素を追加(torch.cat:複数のテンソルを結合)
        label.append(example["label"])
    
    padded["input_ids"] = torch.stack(input_ids) #tensorのリストを結合して1つのtensorにまとめる
    padded["label"] = torch.stack(label)
    
    return padded

if __name__ == "__main__":
    
    print(collate(knock71.train_data[:5]))