import pickle
from collections import Counter


with open("chapter07/data/logistic_regression_sst2.pkl", "rb") as f:
    model = pickle.load(f)


with open("chapter07/data/sst2_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 推論したい文
x = "the worst movie I 've ever seen"

# 特徴量化
x_feature = dict(Counter(x.split()))
X_test = vectorizer.transform([x_feature])

# 予測
pred = model.predict(X_test)[0]
probs = model.predict_proba(X_test)[0]

print("Prediction:")
print("negative" if pred == 0 else "positive")

print("\nProbabilities:")
for label, prob in zip(model.classes_, probs):
    sentiment = "negative" if label == 0 else "positive"
    print(f"P({sentiment}|x) = {prob:.6f}")
