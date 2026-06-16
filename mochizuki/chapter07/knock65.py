from collections import Counter
import joblib

data = joblib.load('model.pkl')
clf, vec = data['clf'], data['vec']

def to_bow(text):
    return dict(Counter(text.lower().split()))

def predict_sentiment(text):
    X = vec.transform([to_bow(text)])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    label = 'positive' if pred == 1 else 'negative'
    confidence = proba[clf.classes_.tolist().index(pred)]
    return label, confidence

test_text = "the worst movie I 've ever seen"
label, conf = predict_sentiment(test_text)
print(f'text:       {test_text}')
print(f'sentiment:  {label} ({conf:.4f})')
