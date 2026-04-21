str = 'パタトクカシーー'
ans = ''

for i in range(0, len(str), 2):
    ans += str[i+1]

print(ans)