def tail_N(file_path: str, N: int):
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines[-N:]:
        print(line.strip())

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    N = 10

    tail_N(file_path, N)
