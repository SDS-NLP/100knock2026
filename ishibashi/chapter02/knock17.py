def unique_data(path):
    """1列目の文字列の異なりを求める関数"""
    data_set = set()

    with open(path) as f:
        for line in f:
            data = line.split('\t')[0]
            data_set.add(data)

    for data in sorted(data_set):
        print(data)

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    unique_data(path)

"""
UNIXコマンド
cut -f 1 popular-names.txt | sort | uniq
"""