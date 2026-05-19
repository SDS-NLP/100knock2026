from janome.tokenizer import Tokenizer

def extract_noun_no_noun():
    text = """メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。"""

    t = Tokenizer()
    
    tokens = list(t.tokenize(text))

    print("【抽出された『名詞＋の＋名詞』一覧】")
    print("-" * 35)

    for i in range(1, len(tokens) - 1):
        
        before = tokens[i - 1]
        current = tokens[i]
        after = tokens[i + 1]

        is_before_noun = before.part_of_speech.split(',')[0] == '名詞'
        is_current_no  = current.surface == 'の'
        is_after_noun  = after.part_of_speech.split(',')[0] == '名詞'


        if is_before_noun and is_current_no and is_after_noun:
            print(before.surface + current.surface + after.surface)

if __name__ == "__main__":
    extract_noun_no_noun()