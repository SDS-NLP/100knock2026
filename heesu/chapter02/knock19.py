def knock19():
    lines = []
    with open("popular-names.txt", "r") as fp:
        for line in fp:
            if line.strip() != "":
                lines.append(line)

    print("".join(sorted(lines, key=lambda x: x.split()[2])), end="")


knock19()
