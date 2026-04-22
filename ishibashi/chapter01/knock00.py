def mix_words(str1, str2):
    """同じ文字数の単語を交互に組み合わせる関数"""
    ans = ''
    
    for i in range(len(str1)):
        ans += str1[i] + str2[i]
    
    return ans

if __name__ == "__main__":
    str1 = 'パトカー'
    str2 = 'タクシー'
    
    print(mix_words(str1, str2))