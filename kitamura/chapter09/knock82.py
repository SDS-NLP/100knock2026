from transformers import pipeline

model_name = "bert-base-uncased"
unmasker = pipeline("fill-mask", model=model_name, top_k=10)

text = "The movie was full of [MASK]."

results = unmasker(text)

for i ,result in enumerate(results, 1):
    word = result["token_str"]
    score = result["score"]
    print(f"{i}：{word} ({score}%)")

"""1：fun (0.10711896419525146%)
2：surprises (0.06634499877691269%)
3：drama (0.044684115797281265%)
4：stars (0.027217146009206772%)
5：laughs (0.025412829592823982%)
6：action (0.019516929984092712%)
7：excitement (0.019038112834095955%)
8：people (0.01829024963080883%)
9：tension (0.015030577778816223%)
10：music (0.014646251685917377%)"""