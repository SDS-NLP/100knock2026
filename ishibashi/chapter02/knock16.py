import random

def shuffle_lines(path):
    """各行をランダムに並び替える関数"""
    with open(path) as f:
        lines = f.readlines()

    random.shuffle(lines)

    for line in lines:
        print(line, end="")

if __name__ == "__main__":
    path = './chapter02/popular-names.txt'

    shuffle_lines(path)

"""
UNIXコマンド
shuf popular-names.txt
"""