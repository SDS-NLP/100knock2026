from collections import Counter

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


def create_dict_list(data):
    instance_list = []
    for i in range(len(data)):
        tokens = data[i][0].split()
        feature_dict = dict(Counter(tokens))

        instance = {
            "text" : data[i][0],
            "label" : data[i][1],
            "feature" : feature_dict
        }

        instance_list.append(instance)
    return instance_list

train_dict_list = create_dict_list(train)
dev_dict_list = create_dict_list(dev)

print(train_dict_list[0])