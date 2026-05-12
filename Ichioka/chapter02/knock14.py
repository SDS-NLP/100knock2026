def head(file_path, n=10):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(line.split()[0])

head("popular-names.txt", 10)