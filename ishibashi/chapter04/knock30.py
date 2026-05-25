import MeCab

def extract_verbs(text):
    """textに含まれる動詞をすべて表示する関数"""
    tagger = MeCab.Tagger()
    result = set()

    parsed_lines = tagger.parse(text).split('\n')
    for line in parsed_lines:
        if line == 'EOS' or line == '':
            continue
        
        parts = line.split('\t')
        pos = parts[4].split('-')[0]
        if pos == '動詞':
            result.add(parts[0])

    return result


if __name__ == "__main__":

    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """

    print(extract_verbs(text))