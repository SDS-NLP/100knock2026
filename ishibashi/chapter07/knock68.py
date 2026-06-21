import os
import joblib

def check_feature_weight():
    model_file = './chapter07/lr_model.joblib'
    vec_file = './chapter07/vectorizer.joblib'

    if not os.path.exists(model_file) or not os.path.exists(vec_file):
        print("必要なファイルのいずれかが見つかりません")
        return
    
    clf = joblib.load(model_file)
    vec = joblib.load(vec_file)

    feature_names = vec.get_feature_names_out()
    weights = clf.coef_[0]

    feature_weights = list(zip(feature_names, weights))

    top_20 = sorted(feature_weights, key=lambda x: x[1], reverse=True)[:20]
    bottom_20 = sorted(feature_weights, key=lambda x: x[1])[:20]

    print("重みの高い特徴量トップ20")
    for i, (name, weight) in enumerate(top_20, 1):
        display_name = f'"{name}"'
        print(f"{i:>2}: {display_name:<20} | 重み: {weight:+.4f}")

    print("重みの低い特徴量トップ20")
    for i, (name, weight) in enumerate(bottom_20, 1):
        display_name = f'"{name}"'
        print(f"{i:>2}: {display_name:<20} | 重み: {weight:+.4f}")

if __name__ == "__main__":
    check_feature_weight()