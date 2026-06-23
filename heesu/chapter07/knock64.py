from knock61 import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

DATA_DIR = "./SST-2"

train = load_dataset(f"{DATA_DIR}/train.tsv")
dev = load_dataset(f"{DATA_DIR}/dev.tsv")

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform([d["feature"] for d in train])
y_train = [d["label"] for d in train]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

X_dev0 = vectorizer.transform([dev[0]["feature"]])
proba = model.predict_proba(X_dev0)[0]

print(f"Text: {dev[0]['text']}")
for label, p in zip(model.classes_, proba):
    print(f"  P(label={label}) = {p:.4f}")
