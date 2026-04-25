file = "popular-names.txt"
count = 0

with open(file, "r", encoding = "utf-8") as f:
    for i in f:
        count += 1

print(count)