def split_file(file_path, n):
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    chunk_size = (total + n - 1) // n  # 切り上げ

    for i in range(n):
        chunk = lines[i * chunk_size : (i + 1) * chunk_size]
        with open(f"part_{i}.txt", "w", encoding="utf-8") as f:
            f.writelines(chunk)

split_file("popular-names.txt", 10)