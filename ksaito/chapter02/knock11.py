def read_lines(file_path):
        with open(file_path) as file:
            for line in file:
                yield line.strip()

def head_N(file_path: str, N: int):
    for count, line in enumerate(read_lines(file_path)):
        if count >= N:
            break
        print(f"count={count},line={line}")

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    N = 10

    head_N(file_path, N)





