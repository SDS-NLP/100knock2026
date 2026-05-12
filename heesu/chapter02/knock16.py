import random


def knock16():
    lines = []
    with open("popular-names.txt", "r") as fp:
        for line in fp:
            lines.append(line)

    random.shuffle(lines)
    print("".join(lines), end="")


knock16()
