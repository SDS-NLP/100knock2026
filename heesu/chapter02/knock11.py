def head(l=5):
    with open("popular-names.txt", "r") as fp:
        for i, line in enumerate(fp):
            if i == l:
                return
            print(line, end="")
            # print(line.strip())


head(10)
