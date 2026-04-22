def n_gram(text, n):
    answer = []
    for i in range(len(text)-n+1):
        answer.append(text[i:i+n])

    return answer

X = n_gram("paraparaparadise", 2)
Y = n_gram("paragraph", 2)

wa =[]
seki =[]
sa = []

for x in X:
    if x in Y:
        if x not in seki:
            seki.append(x)

    else:
        if x not in sa:
            sa.append(x)

wa = seki
for y in Y:
    if y not in wa:
        wa.append(y)

print(wa)
print(sa)
print(seki)
    
word = "se"
if word in X:
    print("'se'はXの要素です")
else:
    print("'se'はXの要素ではありません")

if word in Y:
    print("'se'はYの要素です")

else:
    print("'se'はYの要素ではありません")