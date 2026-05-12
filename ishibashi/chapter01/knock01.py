def get_letters(str):
    """文字列から偶数文字目を取り出す関数"""
    ans = ""

    for i in range(0, len(str), 2):
        ans += str[i+1]
    
    return ans

if __name__ == "__main__":
    str = "パタトクカシーー"
    
    print(get_letters(str))