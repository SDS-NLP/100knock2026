from collections import defaultdict
from operator import itemgetter

from knock11 import read_lines


def build_name_count_map(file_path):
    name_to_count = defaultdict(int)
    for line in read_lines(file_path):
        name = line.split()[0]
        name_to_count[name] += 1

    return sorted(name_to_count.items(), key=itemgetter(1), reverse=True)

if __name__ == "__main__":
    file_path = "data/popular-names.txt"
    print(build_name_count_map(file_path))
