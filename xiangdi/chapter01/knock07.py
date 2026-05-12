def template(x, y, z):
    """テンプレートによる文生成.
    「x時のyはz」という文字列を返す.
    """
    return f"{x}時の{y}は{z}"

result = template(12, "気温", 22.4)
print(result)