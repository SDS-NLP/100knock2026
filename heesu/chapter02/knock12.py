def tail(l=5, filename="popular-names.txt"):
    lines = []
    with open(filename, "r") as fp:
        for line in fp:
            lines.append(line)
    print("".join(lines[-l:]), end="")


tail(10)
