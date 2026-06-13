#Bag of Words (BoW) に基づき、学習データ（train.tsv）および検証データ（dev.tsv）のテキストを特徴ベクトルに変換したい。ここで、ある事例のテキストの特徴ベクトルは、テキスト中に含まれる単語（スペース区切りのトークン）の出現頻度で構成する。例えば、”too loud , too goofy”というテキストに対応する特徴ベクトルは、以下のような辞書オブジェクトで表現される。
#{'too': 2, 'loud': 1, ',': 1, 'goofy': 1}
#各事例はテキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトでまとめておく。例えば、先ほどの”too loud , too goofy”に対してラベル”0”（ネガティブ）が付与された事例は、以下のオブジェクトで表現される。
#{'text': 'too loud , too goofy', 'label': '0', 'feature': {'too': 2, 'loud': 1, ',': 1, 'goofy': 1}}
#学習データと検証データの各事例を上記のような辞書オブジェクトに変換したうえで、学習データと検証データのそれぞれを、辞書オブジェクトのリストとして表現せよ。さらに、学習データの最初の事例について、正しく特徴ベクトルに変換できたか、目視で確認せよ。

import knock60

train_data = knock60.train_data
dev_data = knock60.dev_data

bow_train = []
bow_dev = []

for i in range(len(train_data)):
    
    info = {}
    feature = {} #特徴ベクトルを格納する辞書
    
    info["text"] = train_data[i][0] #データのテキスト
    info["label"] = train_data[i][1] #データのラベル
    
    words = train_data[i][0].split()
    
    for word in words:
        
        if word in feature:
            
            feature[word] += 1
            
        else:
            
            feature[word] = 1
    
    info["feature"] = feature #データの特徴ベクトル
    bow_train.append(info)
    
for i in range(len(dev_data)):
    
    info = {}
    feature = {}
    
    info["text"] = dev_data[i][0]
    info["label"] = dev_data[i][1]
    
    words = dev_data[i][0].split()
    
    for word in words:
        
        if word in feature:
            
            feature[word] += 1
            
        else:
            
            feature[word] = 1
    
    info["feature"] = feature
    bow_dev.append(info)
    
if __name__ == "__main__":
    
    print(bow_train[0])
    print(bow_dev[0])