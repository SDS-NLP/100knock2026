"""
文章のトークン数を計測

事前準備:
python3 -m pip install google-genai
export GEMINI_API_KEY="your-api-key"
"""

import os

from google import genai


MODEL_NAME = "gemini-2.5-flash"
TEXT = """
吾輩は猫である。名前はまだ無い。

どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
""".strip()


api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise SystemExit(
        "GEMINI_API_KEY もしくは GOOGLE_API_KEY を環境変数に設定してください。"
    )

client = genai.Client(api_key=api_key)
result = client.models.count_tokens(
    model=MODEL_NAME,
    contents=TEXT,
)

print("token_count:", result.total_tokens)
