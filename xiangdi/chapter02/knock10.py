count = 0

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    for line in f:
        count += 1
        
print(count)

# wc -l /Users/caitlyn/Downloads/popular-names.txt