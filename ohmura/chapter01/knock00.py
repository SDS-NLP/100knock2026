str1 = "パトカー"
str2 = "タクシー"
result = ""
for s1, s2 in zip(str1, str2):
    result += s1 + s2

print(result)