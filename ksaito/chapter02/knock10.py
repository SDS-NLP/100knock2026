def count_num_lines(file_path: str) -> int:
    with open(file_path, "rb") as file:
        return sum(1 for _ in file)

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    print(count_num_lines(file_path))


