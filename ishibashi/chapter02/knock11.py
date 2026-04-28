def head_N(path, N):
    """先頭からN行を出力する関数"""
    lines_cnt = 0

    with open(path) as f:
        for line in f:
            lines_cnt += 1
            if lines_cnt <= N:
                print(line)
            elif lines_cnt > N:
                break

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'
    N = 10

    head_N(path, N)

"""
UNIXコマンド
head -n 10 popular-names.txt
"""