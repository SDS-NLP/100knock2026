"""
"The movie was full of incomprehensibilities."という文をトークンに分解し、
トークン列を表示するスクリプト
"""

from transformers import BertTokenizer


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"   # 事前学習済みモデル名
TARGET_SENTENCE = "The movie was full of incomprehensibilities."


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    tokens = tokenizer.tokenize(TARGET_SENTENCE)

    print(f"文     : {TARGET_SENTENCE}")
    print(f"トークン数: {len(tokens)}")
    print(f"トークン列: {tokens}")


if __name__ == "__main__":
    main()
