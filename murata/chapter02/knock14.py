with open('popular-names.txt') as f:
    for i, line in enumerate(f):
        print(line.split('\t')[0])

# head -n 10 popular-names.txt | cut -f 1