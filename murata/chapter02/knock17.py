with open('popular-names.txt') as f:
    names = {line.split('\t')[0] for line in f}

for name in sorted(names):
    print(name)
print(f'種類数: {len(names)}')

# cut -f 1 popular-names.txt | sort | uniq