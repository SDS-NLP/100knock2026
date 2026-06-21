import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"

def text_to_feature(text):
    return dict(Counter(text.split()))

def load_sst2_as_dict_list(path):
    df = pd.read_csv(path, sep="\t")

    data = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = str(row["label"])

        data.append({
            "text": text,
            "label": label,
            "feature": text_to_feature(text)
        })

    return data

train_data = load_sst2_as_dict_list(train_path)

X_train_dict = [example["feature"] for example in train_data]
y_train = [int(example["label"]) for example in train_data]

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train_dict)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def predict_sentiment(text):
    feature = text_to_feature(text)
    X = vectorizer.transform([feature])

    pred_label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label_name = "positive" if pred_label == 1 else "negative"

    print("text:", text)
    print("predicted label:", pred_label)
    print("result:", label_name)

    return pred_label

predict_sentiment("the worst movie I 've ever seen")