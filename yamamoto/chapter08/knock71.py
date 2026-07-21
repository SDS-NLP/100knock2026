#General Language Understanding Evaluation (GLUE) ベンチマークで配布されているStanford Sentiment Treebank (SST) をダウンロードし、訓練セット（train.tsv）と開発セット（dev.tsv）のテキストと極性ラベルと読み込み、全てのテキストをトークンID列に変換せよ。このとき、単語埋め込みの語彙でカバーされていない単語は無視し、トークン列に含めないことにせよ。また、テキストの全トークンが単語埋め込みの語彙に含まれておらず、空のトークン列となってしまう事例は、訓練セットおよび開発セットから削除せよ（このため、第7章の実験で得られた正解率と比較できなくなることに注意せよ）。
#事例の表現方法は任意でよいが、例えば”contains no wit , only labored gags”がネガティブに分類される事例は、次のような辞書オブジェクトで表現すればよい。
#{'text': 'contains no wit , only labored gags',
# 'label': tensor([0.]),
# 'input_ids': tensor([ 3475,    87, 15888,    90, 27695, 42637])}
#この例では、textはテキスト、labelは分類ラベル（ポジティブならtensor([1.])、ネガティブならtensor([0.])）、input_idsはテキストのトークン列をID列で表現している。

import csv
import torch
import knock70

train_data = []
dev_data = []

with open("SST-2/train.tsv", "r", encoding = "utf-8") as file: #訓練データ用
    
    reader = csv.reader(file, delimiter = "\t")
    next(reader)
    
    for line in reader:
        
        text_dict = {} #データの1つのテキストについての情報を格納する辞書
        
        text_dict["text"] = line[0] #textはtabで分割した1つめ
        text_dict["label"] = torch.tensor([float(line[1])]) #ラベルはtabで分割した2つめ
    
        text = line[0].split() #textを分割
        
        input_ids = [] #分割された各単語のIDを格納
        
        for token in text:
            
            if token in knock70.word_to_id: #その単語のIDが存在する場合
                
                id_number = knock70.word_to_id[token]
                input_ids.append(id_number)
        
        if len(input_ids) == 0: #全ての単語にIDが存在しなければ飛ばす
            
            continue
        
        text_dict["input_ids"] = torch.tensor(input_ids)
        train_data.append(text_dict)
        
with open("SST-2/dev.tsv", "r", encoding = "utf-8") as file: #開発データ用
    
    reader = csv.reader(file, delimiter = "\t")
    next(reader)
    
    for line in reader:
        
        text_dict = {}
        
        text_dict["text"] = line[0]
        text_dict["label"] = torch.tensor([float(line[1])])
    
        text = line[0].split()
        
        input_ids = []
        
        for token in text:
            
            if token in knock70.word_to_id:
                
                id_number = knock70.word_to_id[token]
                input_ids.append(id_number)
        
        if len(input_ids) == 0:
            
            continue
        
        text_dict["input_ids"] = torch.tensor(input_ids, dtype = torch.long)
        dev_data.append(text_dict)

if __name__ == "__main__":
    
    print(train_data[:2])
    print(dev_data[:2])