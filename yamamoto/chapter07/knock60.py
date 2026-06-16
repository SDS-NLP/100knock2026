#GLUEのウェブサイトからSST-2データセットを取得せよ。学習データ（train.tsv）と検証データ（dev.tsv）のぞれぞれについて、ポジティブ (1) とネガティブ (0) の事例数をカウントせよ。

import csv

train_data = []
dev_data = []

with open("SST-2/train.tsv", "r", encoding = "utf-8") as file:
    
    reader = csv.reader(file, delimiter = "\t")
    next(reader) #1行目を除く
    
    for row in reader:
        
        train_data.append(row)

with open("SST-2/dev.tsv", "r", encoding = "utf-8") as file:
    
    reader = csv.reader(file, delimiter = "\t")
    next(reader)
    
    for row in reader:
        
        dev_data.append(row)
        
train_positive = 0
train_negative = 0

dev_positive = 0
dev_negative = 0

for i in range(len(train_data)):
    
    if train_data[i][1] == "1":
        
        train_positive += 1
    
    else:
        
        train_negative += 1

for i in range(len(dev_data)):
    
    if dev_data[i][1] == "1":
        
        dev_positive += 1
    
    else:
        
        dev_negative += 1

if __name__ == "__main__":

    print("train:", "positive", train_positive, "negative", train_negative)
    print("dev:", "positive", dev_positive, "negative", dev_negative)