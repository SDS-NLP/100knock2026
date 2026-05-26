from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)



class SenryuList(BaseModel):
    senryus: list[str] = Field(description="生成された10個の川柳のリスト")
    # senryusという名前のリストを返すよう指定

theme = "大学生活"
prompt_46 = f"お題「{theme}」で、5・7・5の川柳をちょうど10個作成してください。"

response_46 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt_46,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SenryuList,
        temperature=0.7 # 創作タスクなので少し高めに設定
    )
)


generated_senryus = response_46.parsed.senryus
for i, s in enumerate(generated_senryus, 1):
    print(f"{i}: {s}")




class Evaluation(BaseModel):
    reason: str = Field(description="評価の理由。情景の浮かびやすさやユーモアについて分析してください。")
    score: int = Field(description="1から10までの整数スコア")


class EvaluationResultList(BaseModel):
    results: list[Evaluation]

# 2. 厳密なルーブリック（評価基準）を含めたジャッジプロンプトを作成
prompt_47 = f"""
あなたは川柳のコンテストの厳格な審査員です。
以下の10個の川柳の「面白さ（ユーモア、共感性、情景の浮かびやすさ）」を10段階（1〜10）で評価してください。

【評価基準】
1-3点: 5・7・5のリズムになっていない、または意味が通じない。
4-6点: 意味は通じるが、ひねりがなく平凡である。
7-8点: 情景が目に浮かび、「あるある」と共感できる面白さがある。
9-10点: 言葉遊びが効いている、または予想外の視点があり非常に秀逸。

【対象の川柳】
{generated_senryus}
"""

response_47 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt_47,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EvaluationResultList,
        temperature=0.0 # 評価タスクなのでブレをなくすため0にする
    )
)

# 3. 評価結果の出力
evaluations = response_47.parsed.results
for i, (senryu, eval_data) in enumerate(zip(generated_senryus, evaluations), 1):
    print(f"句{i}: {senryu}")
    print(f"採点: {eval_data.score} / 10")
    print(f"講評: {eval_data.reason}\n")


"""1: 講義中 夢と現実の 狭間かな
2: バイト終え 深夜のレポート 目が冴える
3: サークルで 新しい友 笑顔咲く
4: 単位取り 卒業までの 道遠し
5: 自由とは 責任伴う 大人の証
6: オンライン 画面の向こう 友の顔
7: テスト前 一夜漬けでも 記憶なし
8: 学食の カレーライスが 力くれる
9: 未来へと 希望を胸に 春を待つ
10: ゼミ論で 教授と語る 夢のあと



句1: 講義中 夢と現実の 狭間かな
採点: 7 / 10
講評: 講義中に眠気と戦う学生の情景が目に浮かび、「あるある」と共感できます。「夢と現実の狭間」という表現が秀逸です。

句2: バイト終え 深夜のレポート 目が冴える
採点: 8 / 10
講評: バイト後の疲労とレポート提出のプレッシャーで目が冴えてしまう学生の情景が鮮明に浮かびます。多くの学生が経験する「あるある」な状況で、共感を誘います。

句3: サークルで 新しい友 笑顔咲く
採点: 5 / 10
講評: 情景は浮かびますが、非常にストレートな表現で、特にひねりやユーモアが感じられません。平凡な印象です。

句4: 単位取り 卒業までの 道遠し
採点: 7 / 10
講評: 単位取得の苦労と卒業までの道のりの長さにため息をつく学生の情景が目に浮かびます。多くの学生が共感できる「あるある」な心情を捉えています。

句5: 自由とは 責任伴う 大人の証
採点: 4 / 10
講評: 哲学的な内容で意味は通じますが、川柳特有の情景描写やユーモア、共感性が薄く、格言のようです。

句6: オンライン 画面の向こう 友の顔
採点: 6 / 10
講評: コロナ禍におけるオンラインでの交流という情景は浮かびますが、描写が非常に直接的で、特にひねりや深い共感、ユーモアは感じられません。

句7: テスト前 一夜漬けでも 記憶なし
採点: 8 / 10
講評: テスト前の一夜漬けの虚しさと、その努力が報われない絶望感が鮮やかに伝わってきます。多くの学生が経験する「あるある」な状況で、強い共感を呼びます。

句8: 学食の カレーライスが 力くれる
採点: 7 / 10
講評: 学食のカレーライスという日常的な風景が目に浮かびます。学生生活におけるささやかな喜びや活力を表現しており、共感を誘います。

句9: 未来へと 希望を胸に 春を待つ
採点: 5 / 10
講評: 希望に満ちた未来を待つという情景は浮かびますが、非常に一般的な表現で、川柳としての面白みやひねりが不足しています。

句10: ゼミ論で 教授と語る 夢のあと
採点: 7 / 10
講評: ゼミ論を巡る教授との深い議論の後の余韻や達成感が「夢のあと」という言葉で巧みに表現されており、情景が目に浮かびます。
"""