filename = "popular-names.txt"
N = 10

with open("popular-names.txt", "r") as f:
    lines = f.readlines()

    last_n_lines = lines[-N:]

    for line in last_n_lines:
        print(line.strip())