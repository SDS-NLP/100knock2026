import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

senryu_text = """

"""

judge_prompt = f"""
以下の川柳それぞれについて、面白さを10段階で評価せよ。
評価は1が最も低く、10が最も高い。

{senryu_text}
"""

judge_response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=judge_prompt,
)

print("評価:")
print(judge_response.text)