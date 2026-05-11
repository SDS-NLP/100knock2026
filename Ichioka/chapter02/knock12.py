def tail(file_path, n=10):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[-n:]:
        print(line, end="")

tail("popular-names.txt", 10)