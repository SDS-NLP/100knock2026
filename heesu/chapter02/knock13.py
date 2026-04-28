def knock13():
    lines = []
    with open("popular-names.txt", "r") as fp:
        for idx, line in enumerate(fp):
            lines.append(line.replace("\t", " "))
            if idx == 9:
                break
    print("".join(lines), end="")


knock13()
