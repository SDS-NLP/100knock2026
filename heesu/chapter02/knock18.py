import csv


def knock18():
    strings = {}
    with open("popular-names.txt", "r", encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter="\t")
        for row in reader:
            strings[row[0]] = strings.get(row[0], 0) + 1

    # Sort by count (descending) and print
    for name, count in sorted(strings.items(), key=lambda x: (-x[1], x[0])):
        print(name)


# Run the function
knock18()
