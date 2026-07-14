from transformers import AutoTokenizer

# 80. トークン化
# "The movie was full of incomprehensibilities." をBERTのトークナイザで分解する。
# 設計: AutoTokenizer(81のAutoModelForMaskedLMとクラス解決の流儀を揃える) /
#       bert-base-uncased / ID列経由で [CLS]/[SEP] 込みの「モデルが実際に受け取る形」を見る。
# WordPiece は意味(形態素)ではなく、学習済み語彙への貪欲最長一致で切る。
# ## は「前トークンの続き」の印。uncased なので入力は全て小文字化される。

MODEL_NAME = "bert-base-uncased"


def load_tokenizer(model_name=MODEL_NAME):
    """81以降も同じトークナイザを使うので関数として公開する。"""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    text = "The movie was full of incomprehensibilities."

    # tokenizer(text) はモデル入力用の辞書を返す。input_ids がトークンID列。
    ids = tokenizer(text)["input_ids"]

    # ID → トークン文字列に戻して表示。[CLS]/[SEP] が自動で付いている。
    print(tokenizer.convert_ids_to_tokens(ids))
