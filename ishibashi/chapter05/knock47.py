import os
from dotenv import load_dotenv
from groq import Groq

def evaluate_senryu():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_id = 'llama-3.3-70b-versatile'

    target_senryu = """
    月がきれいな日　夜の散歩がいい　心は平穏
    """
    theme = "月"

    prompt = """
    あなたは厳正でいてユーモアのセンスもある優秀な川柳の審査員です。
    以下に挙げる10個の川柳は「{theme}」をお題としたもので、「面白さ、共感度」を1から10の整数で評価てください。
    また、出力は以下のフォーマットに厳密に従ってください。

    【評価スコア】(1〜10の整数のみ)
    【評価理由】(2〜3文で簡潔に)

    評価対象の川柳: 「{target_senryu}」
    """

    print(f"対象の川柳: {target_senryu}\n")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model_id,
            temperature=0.0,
        )

        response_text = chat_completion.choices[0].message.content
        print(response_text)

    except Exception as e:
        print(f"エラー発生: {e}")

if __name__ == "__main__":
    evaluate_senryu()