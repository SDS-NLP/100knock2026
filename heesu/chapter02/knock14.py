def knock14():
    lines = []
    with open("popular-names.txt", "r") as fp:
        for idx, line in enumerate(fp):
            lines.append(line.split("\t")[0])
            if idx == 9:
                break
    print("\n".join(lines))


knock14()
