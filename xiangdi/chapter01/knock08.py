def cipher(sentence):
    """与えられた文字列の各文字を、以下の仕様で変換する.
    英小文字ならば (219 - 文字コード) のASCIIコードに対応する
    文字に置換.
    その他の文字はそのまま出力
    """
    result = ""
    for a in sentence:
        if 'a' <= a <= 'z':
            result += chr(219 - ord(a))
        else:
            result += a
    return result

sentence = "Hi, I am a cat."
print(cipher(sentence))