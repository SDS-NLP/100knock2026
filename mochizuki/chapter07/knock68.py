import joblib
import numpy as np

data = joblib.load('model.pkl')
clf, vec = data['clf'], data['vec']

feature_names = np.array(vec.feature_names_)
weights = clf.coef_[0]

top_positive = np.argsort(weights)[-20:][::-1]
top_negative = np.argsort(weights)[:20]

print('Top 20 positive features:')
for i in top_positive:
    print(f'  {feature_names[i]:20s}  {weights[i]:+.4f}')

print('\nTop 20 negative features:')
for i in top_negative:
    print(f'  {feature_names[i]:20s}  {weights[i]:+.4f}')
