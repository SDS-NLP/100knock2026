path = "popular-names.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
N = int(input("N:"))
length = int(len(lines)/N)
for i in range(N):
    start = i*length
    end = (i+1)*length
    contents = lines[start:end]
    file_name = f"file_{i}"
    with open(file_name, "w", encoding="utf-8") as ff:
        ff.writelines(contents)