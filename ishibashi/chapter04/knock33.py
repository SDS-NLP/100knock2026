import spacy

def extract_dependency_pairs(text):
    """係り受け解析を適用し、係り元と係り先のトークンをタブ区切り形式ですべて抽出する関数"""
    nlp = spacy.load("ja_ginza")
    doc = nlp(text)

    pairs = []
    for token in doc:
       if token.is_space:
           continue
       
       pairs.append((token.text, token.head.text))

    return pairs

if __name__ == "__main__":

    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """

    for source, target in extract_dependency_pairs(text):
        print(f"{source}\t{target}")