def template():
    x, y, z = input("x、y、zを入力してください: ").split()
    return f"{x}の時の{y}は{z}"


def main():
    print(template())


main()
