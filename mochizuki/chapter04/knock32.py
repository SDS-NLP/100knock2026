import MeCab

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

tagger = MeCab.Tagger('-r /opt/homebrew/etc/mecabrc')
node = tagger.parseToNode(text)
tokens = []
while node:
    if node.surface:
        tokens.append((node.surface, node.feature.split(',')))
    node = node.next

phrases = []
for i in range(len(tokens) - 2):
    s0, f0 = tokens[i]
    s1, f1 = tokens[i + 1]
    s2, f2 = tokens[i + 2]
    if f0[0] == '名詞' and s1 == 'の' and f2[0] == '名詞':
        phrases.append(f'{s0}の{s2}')

if __name__ == "__main__":
    for phrase in phrases:
        print(phrase)
