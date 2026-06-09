from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

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
    features = []
    labels = []
    for i in range(len(data)):
        tokens = data[i][0].split()
        feature_dict = dict(Counter(tokens))
        features.append(feature_dict)
        labels.append(data[i][1])
    return features, labels  


x_train_dict, y_train = create_dict_list(train)
x_dev_dict, y_dev = create_dict_list(dev)

vectorizer = DictVectorizer(sparse=True)
x_train = vectorizer.fit_transform(x_train_dict)
x_dev = vectorizer.transform(x_dev_dict)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_dev)

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

words_weights = list(zip(feature_names, weights))
words_weights.sort(key=lambda x: x[1])

print("worst10")
for word, weight in words_weights[:10]:
    print(f"{word} : {weight}")

print("top10")
for word, weight in reversed(words_weights[-10:]):
    print(f"{word} : {weight}")

"""
worst10
lacking : -4.283609111128958
worst : -4.042351050462471
lacks : -4.014681069670004
devoid : -3.6200240982017915
mess : -3.5396902031361845
failure : -3.5034820705396075
stupid : -3.30051081108569
bore : -3.2129292423367253
flat : -3.2105815019662898
waste : -3.1382794180752165

top10
refreshing : 3.3931761837619914
remarkable : 3.359958932187323
powerful : 3.2024335486788793
hilarious : 3.1376287773178637
beautiful : 2.9787185331296726
wonderful : 2.953844962832077
prose : 2.844533336290969
appealing : 2.833230732130623
terrific : 2.817428223187838
treat : 2.785027539718062"""