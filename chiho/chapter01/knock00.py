#00.パタトクカシー
#2つの文字列「パトカー」と「タクシー」の文字を先頭から交互に連結し、文字列「パタトクカシーー」を得よ。

text1 = "パトカー"
text2 = "タクシー"
x = ""

for i in range(len(text1)):
    x += text1[i]
    x += text2[i]

print(x)