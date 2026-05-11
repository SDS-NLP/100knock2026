#03. 円周率
#“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”という文を単語に分解し、各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ。

raw_text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
text = raw_text.replace(".", "").replace(",", "")

lengths = [len(i) for i in text.split()]
print(lengths)
