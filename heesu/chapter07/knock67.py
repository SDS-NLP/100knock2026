from knock61 import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "./SST-2"

train = load_dataset(f"{DATA_DIR}/train.tsv")
dev = load_dataset(f"{DATA_DIR}/dev.tsv")

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform([d["feature"] for d in train])
y_train = [d["label"] for d in train]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

X_dev = vectorizer.transform([d["feature"] for d in dev])
y_dev = [d["label"] for d in dev]

y_train_pred = model.predict(X_train)
y_dev_pred = model.predict(X_dev)

for split_name, y_true, y_pred in [("Train", y_train, y_train_pred), ("Dev", y_dev, y_dev_pred)]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="1")
    rec = recall_score(y_true, y_pred, pos_label="1")
    f1 = f1_score(y_true, y_pred, pos_label="1")
    print(f"{split_name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
