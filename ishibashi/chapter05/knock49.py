import os
from dotenv import load_dotenv
from groq import Groq

def count_tokens():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_id = 'llama-3.3-70b-versatile'

    text = """
    吾輩は猫である。名前はまだ無い。

    どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text
                }
            ],
            model=model_id,
            max_tokens=1,
        )

        prompt_tokens = chat_completion.usage.prompt_tokens

        print(f"テキストの文字数: {len(text)}文字")
        print(f"Llama3でのトークン数: 約{prompt_tokens}トークン")

    except Exception as e:
        print(f"エラー発生: {e}")

if __name__ == "__main__":
    count_tokens()