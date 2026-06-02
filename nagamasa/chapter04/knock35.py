from spacy import displacy
import spacy

# 35. 係り受け木

text = "メロスは激怒した。"
nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
doc = nlp(text)

# ファイルに保存
html = displacy.render(doc, style="dep", page=True)
with open("dep.html", "w") as f:
    f.write(html)

# displacy.render()はHTML文字列を返す
# page=TrueでHTMLページ全体として出力する
# style="dep"は係り受け、style="ent"は固有表現抽出

# displacy.serve()はローカルサーバーを起動してブラウザで表示する
# ブロッキング処理なので次の行に進まない
# ファイル保存と併用する場合はserve()を後に書くか削除する

# text.strip()で前後の改行を除去する
# \nが含まれると余分なトークンが生成される

# PROPNは固有名詞、VERBは動詞、ADPは助詞、AUXは助動詞
# nsubjは主語、caseは格助詞、auxは助動詞の係り受けラベル