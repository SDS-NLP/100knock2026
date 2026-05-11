str1 = "パトカー"
str2 = "タクシー"
ans = ""

for i in range(len(str1)):
    ans += str1[i]
    ans += str2[i]

if __name__ == "__main__":
    print(ans)