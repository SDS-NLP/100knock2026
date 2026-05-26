import os
import time
from dotenv import load_dotenv
from groq import Groq

def run_multi_turn_dialogue():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_id = 'llama-3.3-70b-versatile'

    prompt_44 = """
    つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
    """

    prompt_45 = """
    さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
    """

    messages = [
        {
            "role": "system",
            "content": "あなたは優秀なAIアシスタントです。以下の文章を注意深く読み、論理的に答えてください"
        }
    ]

    try:
        print("問44の応答")

        messages.append(
            {
                "role": "user",
                "content": prompt_44
            }
        )

        response_1 = client.chat.completions.create(
            messages=messages,
            model=model_id,
            temperature=0.0,
        )

        assistant_reply_1 = response_1.choices[0].message.content
        print(f"{assistant_reply_1}")

        messages.append(
            {
                "role": "assistant",
                "content": assistant_reply_1
            }
        )

        time.sleep(3)

        print("-" * 40)
        print("問45の応答")

        messages.append(
            {
                "role": "user",
                "content": prompt_45
            }
        )

        response_2 = client.chat.completions.create(
            messages=messages,
            model=model_id,
            temperature=0.0,
        )

        assistant_reply_2 = response_2.choices[0].message.content
        print(f"{assistant_reply_2}")
        
    except Exception as e:
        print(f"エラー発生: {e}")

if __name__ == "__main__":
    run_multi_turn_dialogue()