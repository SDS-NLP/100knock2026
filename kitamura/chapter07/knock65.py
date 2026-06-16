from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train = []
with open("./SST-2/train.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        train.append(line.strip().split("\t"))


def create_dict_list(data):
    features = []
    labels = []
    for i in range(len(data)):
        tokens = data[i][0].split()
        feature_dict = dict(Counter(tokens))
        features.append(feature_dict)
        labels.append(data[i][1])
    return features, labels  

def create_test_dict(sentence):
    features = []
    tokens = sentence.split()
    feature_dict = dict(Counter(tokens))
    features.append(feature_dict)
    return features


x_train_dict, y_train = create_dict_list(train)

vectorizer = DictVectorizer(sparse=True)
x_train = vectorizer.fit_transform(x_train_dict)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

target_text = input(f"please write a sentence : ")
target_dict = create_test_dict(target_text)
x_target = vectorizer.transform(target_dict)
pred_label = model.predict(x_target)

print(pred_label)

# please write a sentence : the worst movie I ‘ve ever seen
# ['0']