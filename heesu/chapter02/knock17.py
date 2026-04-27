import csv


def knock17():
    strings = set()
    with open("popular-names.txt", "r") as fp:
        reader = csv.reader(fp, delimiter="\t")
        for row in reader:
            strings.add(row[0])

    print("\n".join(sorted(strings)))


knock17()
