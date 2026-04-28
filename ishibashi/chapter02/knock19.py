def sort_by_third_column(path):
    """3列目の数値で降順に並び替える関数"""
    with open(path) as f:
        lines = f.readlines()

    sorted_lines = sorted(lines, key=lambda x: int(x.split('\t')[2]), reverse=True)

    for line in sorted_lines:
        print(line, end="")

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    sort_by_third_column(path)

"""
UNIXコマンド
sort -k 3 -nr popular-names.txt
"""