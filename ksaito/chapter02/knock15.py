import math


def N_split(file_path, N):
    with open(file_path) as f:
        lines = f.readlines()
    chunk = math.ceil(len(lines) / N)

    for i in range(N):
        with open(f"data/sub_pop_names_{i}.txt", "w") as f:
            f.writelines(lines[i*chunk: (i + 1)*chunk])

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    N = 10

    N_split(file_path, N)

