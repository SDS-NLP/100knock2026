from collections import Counter

with open('popular-names.txt') as f:
    counter = Counter(line.split('\t')[0] for line in f)

for name, count in counter.most_common():
    print(f'{count}\t{name}')
    
# cut -f 1 popular-names.txt | sort | uniq -c | sort -rn
