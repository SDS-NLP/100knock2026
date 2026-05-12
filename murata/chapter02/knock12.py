from collections import deque

with open('popular-names.txt') as f:
    for line in deque(f, 10):
        print(line, end='')

# tail -n 10 popular-names.txt