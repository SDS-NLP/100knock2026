import os
from dotenv import load_dotenv
from groq import Groq

def run_dialogue():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_id = 'llama-3.3-70b-versatile'

    prompt = """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt,
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
    run_dialogue()