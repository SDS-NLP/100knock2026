N = 10
filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    lines = f.readlines()

total = len(lines)
chunk_size = total // N
remainder = total % N

start = 0
for i in range(N):
    end = start + chunk_size + (1 if i < remainder else 0)

    with open(f"out_{i}.txt", "w") as out:
        out.writelines(lines[start:end])

    start = end

# split -n 10 shiratori/chapter02/popular-names.txt
