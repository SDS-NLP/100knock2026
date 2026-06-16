import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def convert_to_bow(file_path):
    df = pd.read_csv(file_path, sep='\t')
    dataset = []
    for _, row in df.iterrows():
        text = row['sentence']
        label = str(row['label'])
        feature = dict(Counter(text.split()))
        dataset.append({'text': text, 'label': label, 'feature': feature})
    return dataset

train_data = convert_to_bow('SST-2/train.tsv')
dev_data = convert_to_bow('SST-2/dev.tsv')

X_train_dict = [d['feature'] for d in train_data]
y_train = [d['label'] for d in train_data]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

X_dev_dict = [d['feature'] for d in dev_data]
X_dev = vec.transform(X_dev_dict)
y_dev = [d['label'] for d in dev_data]

first_dev_instance = X_dev[0]
actual_label = y_dev[0]
text = dev_data[0]['text']

predicted_label = clf.predict(first_dev_instance)[0]

print("【検証データの先頭事例に対する予測結果】")
print(f"テキスト : {text}")
print(f"正解ラベル: {actual_label} （{ 'ポジティブ' if actual_label == '1' else 'ネガティブ' }）")
print(f"予測ラベル: {predicted_label} （{ 'ポジティブ' if predicted_label == '1' else 'ネガティブ' }）")
print("-" * 40)
if actual_label == predicted_label:
    print("結果: ✅ 予測は正解と一致しています！")
else:
    print("結果: ❌ 予測は正解と一致していません。")