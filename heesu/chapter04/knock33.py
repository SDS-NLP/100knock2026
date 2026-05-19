import ginza
import spacy

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def extract_dependencies_tsv(text):
    config = {
        "components": {
            "compound_splitter": {
                "split_mode": "A",
            }
        }
    }

    nlp = spacy.load("ja_ginza", config=config)
    doc = nlp(text)

    results = []

    for sent in doc.sents:
        for span in ginza.bunsetu_spans(sent):
            span_root = span.root
            head_token = span_root.head

            head_span = ginza.bunsetu_span(head_token)

            if span != head_span:
                results.append(f"{span.text}\t{head_span.text}")

    return "\n".join(results)


if __name__ == "__main__":
    tsv_output = extract_dependencies_tsv(text)
    print(tsv_output)
