import MeCab

def extract_a_no_b(text):
    """2つの名詞が「の」で連結されている名詞句を抽出する関数"""
    tagger = MeCab.Tagger()
    morphs = []

    parsed_lines = tagger.parse(text).split('\n')
    for line in parsed_lines:
        if line == 'EOS' or line == '':
            continue

        parts = line.split('\t')
        morphs.append({
            'surface': parts[0],
            'parts': parts[4].split('-')[0]
        })

        a_no_b = []
        for i in range(1, len(morphs) - 1):
            if morphs[i]['surface'] == 'の':
                if morphs[i - 1]['parts'] == '名詞' and morphs[i + 1]['parts'] == '名詞':
                    phrase = morphs[i - 1]['surface'] + morphs[i]['surface'] + morphs[i + 1]['surface']
                    a_no_b.append(phrase)
        
    return a_no_b

if __name__ == "__main__":

    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """

    print(extract_a_no_b(text))