import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def train_model():
    # 学習データ(train.tsv)の読み込み
    df = pd.read_csv('SST-2/train.tsv', sep='\t')
    
    # テキストを単語の出現回数（辞書）に変換
    X_train_dict = [dict(Counter(str(row['sentence']).split())) for _, row in df.iterrows()]
    y_train = [str(row['label']) for _, row in df.iterrows()]
    
    # DictVectorizerで辞書のリストを数値の行列（ベクトル）に変換
    vec = DictVectorizer()
    X_train = vec.fit_transform(X_train_dict)
    
    # ロジスティック回帰モデルを学習
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 学習済みの変換器(vec)と分類器(clf)を返す
    return vec, clf

# 学習を実行し、モデルを変数に格納
vec, clf = train_model()

def predict_sentiment(text):
    # 入力された新しいテキストを単語に分割し、辞書化
    feature = dict(Counter(text.split()))
    X_input = vec.transform([feature])
    
    # 学習済みの語彙(vec)を使って、入力をベクトルに変換
    predicted_label = clf.predict(X_input)[0]
    probabilities = clf.predict_proba(X_input)[0]
    
    label_name = "ポジティブ" if predicted_label == '1' else "ネガティブ"
    
    print(f"入力テキスト: {text}")
    print(f"予測結果: {predicted_label} ({label_name})")
    
    for cls, prob in zip(clf.classes_, probabilities):
        name = "ポジティブ" if cls == '1' else "ネガティブ"
        print(f" - {name}の確率: {prob*100:.2f}%")
    print("-" * 40)

sample_text = "the worst movie I 've ever seen"
predict_sentiment(sample_text)