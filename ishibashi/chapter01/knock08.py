def cipher(target):
    """
    以下の仕様で、文字列各文字を変換する関数

    英小文字ならば(219 - 文字コード)のASCIIコードに対応する文字に置換
    その他の文字はそのまま出力
    """
    res = ""

    for i in range(len(target)):
        code = ord(target[i])
        if 97 <= code <= 122:
            code = 219 - code
        res += chr(code)
    
    return res

if __name__ == "__main__":
    target = "Hitotsubashi University"
    res = cipher(target)

    print(f"暗号化: {res}")
    print(f"復号化: {cipher(res)}")