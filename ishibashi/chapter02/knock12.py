def tail_N(path, N):
    """末尾からN行を出力する関数"""
    with open(path) as f:
        lines = f.readlines()
    
    for line in lines[-N:]:
        print(line)

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'
    N = 10

    tail_N(path, 10)

"""
UNIXコマンド
tail -n 10 popular-names.txt
"""