import llm

theme = "時間"

# Step 1: Generate 10 senryu
gen_prompt = f"""お題「{theme}」で川柳を10個作成してください。"""

print(f"お題：{theme}\n")
senryu_result = llm.chat(
    gen_prompt,
    system="あなたは川柳の名人です。ユーモアあふれる川柳を作ります。",
    temperature=0.9,
)
print("=== 生成された川柳 ===")
print(senryu_result)

# Step 2: Evaluate each senryu on a scale of 1–10
eval_prompt = f"""以下の川柳を1つずつ10点満点で評価してください。

川柳リスト：
{senryu_result}
"""

print("\n=== LLMによる評価 ===")
evaluation = llm.chat(
    eval_prompt,
    temperature=0.3,
)
print(evaluation)
