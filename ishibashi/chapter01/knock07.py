def template(x, y, z):
    """引数x,y,zを受け取り、それに合わせて文字列を返す関数"""
    return f"{x}時の{y}は{z}"

if __name__ == "__main__":
    x = 12
    y = "気温"
    z = 22.4

    print(template(x, y, z))