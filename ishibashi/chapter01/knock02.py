def reverse_word(str):
    "文字列を逆に並べる関数"
    ans = str[::-1]
    
    return ans

if __name__ == "__main__":
    str = "stressed"
    
    print(reverse_word(str))