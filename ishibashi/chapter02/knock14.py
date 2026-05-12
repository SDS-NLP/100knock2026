def get_first_column(path):
    """先頭10行の1列目を出力する関数"""
    with open(path) as f:
        for i, line in enumerate(f):
            if i < 10:
                print(line.split('\t')[0])
            else:
                break

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    get_first_column(path)

"""
UNIXコマンド
head -n 10 popular-names.txt | cut -f 1
"""