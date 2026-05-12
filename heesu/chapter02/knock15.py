def knock15(n=10):
    with open("popular-names.txt", "rb") as fp:
        lines = fp.readlines()

    lc = len(lines)
    # Ceiling Operation
    lc_per_file = (lc + n - 1) // n

    for i in range(10):
        with open(f"py_xa{chr(ord('a') + i)}", "wb") as fp:
            fp.write(
                b"".join(
                    lines[i * lc_per_file : min((i + 1) * (lc_per_file), len(lines))]
                )
            )


knock15()
