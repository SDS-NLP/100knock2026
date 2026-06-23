import os
import joblib
from collections import Counter

def predict_posinega_text(text):
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    if not os.path.exists(model_file) or not os.path.exists(vec_file):
        print("エラー: モデルなどが見つかりません")
        return
    
    clf = joblib.load(model_file)
    vec = joblib.load(vec_file)

    feature_dict = dict(Counter(text.split(' ')))
    X_input = vec.transform([feature_dict])

    pred_label = clf.predict(X_input)[0]
    probabilities = clf.predict_proba(X_input)[0]
    classes = clf.classes_

    neg_prob = probabilities[0] if classes[0] == '0' else probabilities[1]
    pos_prob = probabilities[1] if classes[1] == '1' else probabilities[0]

    print(f"判定結果: {'POSI' if pred_label == '1' else 'NEGA'}")
    print(f"詳細確率: Positive: {pos_prob*100:.2f}% / Negative: {neg_prob*100:.2f}%")

def main():
    target_text = "the worst movie I ‘ve ever seen"
    predict_posinega_text(target_text)

if __name__ == "__main__":
    main()