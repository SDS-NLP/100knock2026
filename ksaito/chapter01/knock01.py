def pick_strs(s: str, indexes: list[int]) -> str:
    picked = []
    for i in indexes:
        if not (1 <= i <= len(s)):
            continue
        picked.append(s[i - 1]) # 0-based
        
    return "".join(picked)

if __name__ == "__main__":
    s = "パタトクカシーー"
    indexes = [2, 4, 6, 8] # 1-based
    print(pick_strs(s, indexes))
