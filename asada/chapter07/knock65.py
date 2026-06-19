from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from knock61 import objectify

train_data = objectify("SST-2/train.tsv")

model_logistic = LogisticRegression(max_iter=1000)

features = [item["feature"] for item in train_data]
y = [item["label"] for item in train_data]

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(features)

model_logistic.fit(X, y)


def text_processor(text: str):
    vec = {}
    words = text.split()
    for w in words:
        if w not in vec:
            vec[w] = 1
        else:
            vec[w] += 1
    return vec


text_feature = text_processor("the worst movie I 've ever seen")

X = vec.transform([text_feature])

predicted_label = model_logistic.predict(X)

print(f"予測されたラベル: {predicted_label}")
