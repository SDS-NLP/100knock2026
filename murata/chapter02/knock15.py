import math

with open('popular-names.txt') as f:
    lines = f.readlines()
    N = sum(1 for _ in f)

size = math.ceil(len(lines) / N)
for i in range(N):
    chunk = lines[i * size:(i + 1) * size]
    with open(f'split_{i:02d}.txt', 'w') as fout:
        fout.writelines(chunk)
        
# split -n l/10 -d popular-names.txt split_