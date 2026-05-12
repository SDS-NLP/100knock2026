import random


def shuffle_lines(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        print(line.strip())

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    shuffle_lines(file_path)


