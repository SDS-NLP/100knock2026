p = "パトカー"
t = "タクシー"

print("".join([p[i] + t[i] for i in range(min(len(p), len(t)))]))

# method2
# zip object yields n-length tuples, where n is the number of iterables
# zip関数の返り値はzipオブジェクト. 繰り返し処理が適用された場合に(zip関数を呼び出したときの)引数の数と同じ長さのタプルを生成.
print("".join([j + k for j, k in zip(p, t)]))
