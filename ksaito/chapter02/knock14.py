from knock11 import read_lines


def first_col(file_path, N):
    for count, line in enumerate(read_lines(file_path)):
        if count >= N:
            break
        print(f"Row Count = {count}, Name = {line.split()[0]}")

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    N = 10

    first_col(file_path, N)
