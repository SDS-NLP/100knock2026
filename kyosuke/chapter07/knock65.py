import joblib
from collections import Counter

model = joblib.load('lr_model.joblib')
vec = joblib.load('vectorizer.joblib')

def predict_sentiment(text):
    feature = dict(Counter(text.split()))
    X_input = vec.transform([feature])
    
    pred_label = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]
    prob_negative = probabilities[0]
    prob_positive = probabilities[1]

    sentiment = "ポジティブ" if pred_label == 1 else "ネガティブ"
    confidence = prob_positive if pred_label == 1 else prob_negative
    
    return sentiment, confidence * 100

text = "the worst movie I 've ever seen"
sentiment, confidence = predict_sentiment(text)
print(f"予測結果：{sentiment} 自信度:{confidence:.1f}%")