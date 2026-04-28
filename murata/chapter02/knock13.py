with open('popular-names.txt') as f:
    for i, line in enumerate(f):
        print(line.replace('\t', ' '), end='')
        
# head -n 10 popular-names.txt | tr '\t' ' '