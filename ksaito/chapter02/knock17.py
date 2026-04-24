from knock11 import read_lines


def count_uniq_names(file_path):
    uniq_names = set()
    for line in read_lines(file_path):
        uniq_names.add(line.split()[0])

    return len(uniq_names), uniq_names

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    print(count_uniq_names(file_path))
