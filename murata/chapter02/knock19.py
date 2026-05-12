with open('popular-names.txt') as f:
    lines = [line.rstrip('\n').split('\t') for line in f]

lines.sort(key=lambda x: int(x[2]), reverse=True)

for cols in lines:
    print('\t'.join(cols))

# sort -k 3 -nr popular-names.txt