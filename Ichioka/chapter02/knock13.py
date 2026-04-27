def replace_tab(file_path, n=10):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            replaced_lines = line.replace("\t", " ")
            print(replaced_lines, end="")

replace_tab("popular-names.txt", 10)