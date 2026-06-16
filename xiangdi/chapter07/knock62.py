import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

train_json_path = "/Users/caitlyn/Downloads/SST-2/train.json"

with open(train_json_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

X_train_dict = [example["feature"] for example in train_data]
y_train = [int(example["label"]) for example in train_data]

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train_dict)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)