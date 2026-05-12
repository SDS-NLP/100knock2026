with open("popular-names.txt", "r") as fp:
    buf = fp.read()
    print(buf.count("\n"))
