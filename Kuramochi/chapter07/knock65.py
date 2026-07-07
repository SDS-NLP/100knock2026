"テキストデータを定義→辞書形式に変換→ベクトル化→学習済みモデルで予測（0: ネガティブ、1: ポジティブ）"

import joblib
from collections import Counter


# テキストデータを受け取って、BoW特長料を辞書型で返す関数
def text_to_bow(text) -> dict:
    tokens       = text.split()
    count_tokens = Counter(tokens)

    output_dict = dict(text = text, feature = count_tokens)
    
    return output_dict

# 学習済みモデルとベクトライザーの読み込み
loaded_model      = joblib.load('sst2_lr_model.pkl')
loaded_vectorizer = joblib.load('sst2_vectorizer.pkl')

# 新しいテキストデータ→辞書形式に変換
sample_texts = "the worst movie I have ever seen"
dict_bow     = text_to_bow(sample_texts)

X_test_matrix = loaded_vectorizer.transform([dict_bow['feature']])
prediction    = loaded_model.predict(X_test_matrix)
print(prediction)