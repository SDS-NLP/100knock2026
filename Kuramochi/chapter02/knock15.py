n_parts = 10
with open("popular-names.txt", encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
base  = total // n_parts
extra = total % n_parts
start = 0

for i in range(n_parts):
    end = start + base + (1 if i < extra else 0) # あまりの分配
    with open(f"popular-names-part-{i+1:02}.txt", "w", encoding="utf-8") as out_f:
        out_f.writelines(lines[start:end])
    start = end

# https://qiita.com/b-mente/items/0a57e65687d67b4ac582
# split -n l/10 popular-names.txt