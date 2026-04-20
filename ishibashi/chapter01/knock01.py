str = 'パタトクカシーー'
str_new = ''

for i in range(0, len(str), 2):
    str_new += str[i+1]

print(str_new)