string = "パタトクカシーー"
print("".join([letter for i, letter in enumerate(string) if i % 2 == 1]))

# method2
print(string[1::2])
