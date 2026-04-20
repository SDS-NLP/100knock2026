s1 = "パトカー"
s2 = "タクシー"
result = ""

for i in range(min(len(s1), len(s2))):
    result += s1[i] + s2[i]

print(result)