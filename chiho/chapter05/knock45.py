"""
Gemini APIでマルチターン対話を行う。

事前準備:
    python3 -m pip install google-genai
    export GEMINI_API_KEY="your-api-key"
"""

import os

from google import genai
from google.genai import errors


MODEL_NAME = "gemini-2.5-flash"

first_prompt = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
"""

second_prompt = """
さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
"""


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY もしくは GOOGLE_API_KEY を環境変数に設定してください。"
        )

    try:
        with genai.Client(api_key=api_key) as client:
            chat = client.chats.create(model=MODEL_NAME)
            print("対話を始めます。終了するには exit または quit と入力してください。")

            while True:
                user_message = input("You: ").strip()

                if not user_message:
                    continue

                if user_message.lower() in {"exit", "quit"}:
                    print("対話を終了します。")
                    break

                response = chat.send_message(user_message)
                print("Gemini:", response.text)
    except errors.APIError as error:
        raise SystemExit(f"Gemini API error ({error.code}): {error.message}") from error


if __name__ == "__main__":
    main()

# 実行例

"""対話を始めます。終了するには exit または quit と入力してください。
You: つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
Gemini: つばめちゃんの目的地の駅は **緑が丘** です。

**理由:**

1.  東急大井町線、自由が丘駅で大井町方面の急行に乗車した場合、自由が丘の次の急行停車駅は **大岡山駅** です。
2.  大岡山駅から反対方向（二子玉川方面）へ一駅戻ると、**緑が丘駅** になります。
You: さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
Gemini: この場合のルートを考えましょう。

1.  **自由が丘の次の急行停車駅（反対方向）で降車:**
    東急大井町線で自由が丘駅から「反対方向」、つまり二子玉川方面の急行に乗車した場合、自由が丘の次の急行停車駅は **二子玉川駅** です。
    (自由が丘駅と二子玉川駅の間には、九品仏、尾山台、等々力、上野毛の各駅がありますが、これらは急行停車駅ではありません。)

2.  **二子玉川駅から目的地へ向かう:**
    つばめちゃんの目的地は、最初のケースで特定された **緑が丘駅** です。
    二子玉川駅から反対方向（大井町方面）の「各駅停車」に乗車し、緑が丘駅を目指します。

3.  **何駅先の駅で降りれば良いか:**
    二子玉川駅から大井町方面の各駅停車に乗車し、緑が丘駅までの停車駅を数えます。

    *   上野毛 (1駅目)
    *   等々力 (2駅目)
    *   尾山台 (3駅目)
    *   九品仏 (4駅目)
    *   自由が丘 (5駅目)
    *   **緑が丘** (6駅目)

したがって、二子玉川駅から**6駅先**の駅で降りれば、目的地の緑が丘駅に到着します。
"""