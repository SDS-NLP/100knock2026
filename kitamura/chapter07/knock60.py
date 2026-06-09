import os
import urllib.request
import zipfile

url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
zip_filename = "SST-2.zip"
extract_dir = "./" 
target_file = "./SST-2/train.tsv" 

if not os.path.exists(target_file):
    urllib.request.urlretrieve(url, zip_filename)
    print("ダウンロード完了")

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("解凍完了。")

    if os.path.exists(zip_filename):
        os.remove(zip_filename)
  

    print("SSTダウンロード終了")

else:
    print("すでにダウンロードされています")

train = []
with open("./SST-2/train.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        train.append(line.strip().split("\t"))

dev = []
with open("./SST-2/dev.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        dev.append(line.strip().split("\t"))


def nega_posi_count(data):
    nega = 0
    posi = 0
    for i in range(len(data)):
        if data[i][1] == "0":
            nega += 1

        elif data[i][1] == "1":
            posi += 1

        else:
            print(f"{i} : {data[i][1]}")

    return nega, posi

nega_train, posi_train = nega_posi_count(train)
nega_dev, posi_dev = nega_posi_count(dev)

print(f"negative_train : {nega_train}")
print(f"positive_train : {posi_train}")
print(f"negative_test : {nega_dev}")
print(f"positive_test : {posi_dev}")

"""negative_train : 29780
positive_train : 37569
negative_test : 428
positive_test : 444"""