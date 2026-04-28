from knock11 import read_lines


def tab_to_space(file_path, N):
    for count, line in enumerate(read_lines(file_path)):
        if count >= N:
            break
        print(f"count={count},line={line.replace("\t", " ")}")

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    N = 10

    tab_to_space(file_path, N)

