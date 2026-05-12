def sort_by_f3(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    for l in sorted(lines, key=lambda line: int(line.split()[3]), reverse=True):
        print(l.strip())

if __name__=="__main__":
    file_path = "data/popular-names.txt"
    print(sort_by_f3("data/popular-names.txt"))
