def replace_tab(path):
    """先頭10行のタブ1文字をスペース1文字に置換する関数"""
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 10:
                print(line.replace('\t', ' '), end="")
            else:
                break

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    replace_tab(path)

"""
UNIXコマンド
head -n 10 popular-names.txt | tr '\t' ' '
"""