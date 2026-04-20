def build_sentence(x, y, z):
    return f"{x}時の{y}は{z}"

def main():
    x, y, z = 12, "気温", 22.4
    print(build_sentence(x, y, z))

if __name__ == "__main__":
    main()

