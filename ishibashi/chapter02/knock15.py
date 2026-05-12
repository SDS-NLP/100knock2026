def split_file(path, N):
    """ファイルをN個に分割して保存する関数"""
    with open(path) as f:
        lines = f.readline()

        check_size = len(lines) // N

        for i in range(N):
            start = i * check_size
            if i < N - 1:
                end = (i + 1) * check_size
            else:
                end = len(lines)
            
            with open(f"split_{i}.txt", "w") as out:
                out.writelines(lines[start:end])

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    split_file(path, 10)

""""
UNIXコマンド
split -n l/10 popular-names.txt split_
"""