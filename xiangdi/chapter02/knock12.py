from collections import deque

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    tail = deque(f, maxlen=10)
    
for line in tail:
    print(line, end="")

# tail -n 10 /Users/caitlyn/Downloads/popular-names.txt