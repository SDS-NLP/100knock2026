from janome.tokenizer import Tokenizer

def extract_verbs_with_base_form():
    text = """メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。"""

    t = Tokenizer()
    
    print(f"{'文章中':<12}{'原形':<12}")
    print("-" * 30)

    for token in t.tokenize(text):
        part_of_speech = token.part_of_speech.split(',')[0]
        
        if part_of_speech == '動詞':
            print(f"{token.surface:<15}{token.base_form:<12}")

if __name__ == "__main__":
    extract_verbs_with_base_form()