def generate_sentence(x, y, z):
    return f"{x}時の{y}は{z}"

x = 12
y = "気温"
z = 22.4

result = generate_sentence(x, y, z)
print(result)