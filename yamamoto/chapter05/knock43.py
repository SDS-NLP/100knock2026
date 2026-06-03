#問題42において、実験設定を変化させると正解率が変化するかどうかを調べよ。実験設定の例としては、大規模言語モデルの温度パラメータ、プロンプト、多肢選択肢の順番、多肢選択肢の記号などが考えられる。正解の選択肢を全てDに入れ替えて解答させる例。

import csv
from google import genai
import gemini_api
import time
from google.genai.errors import (
    ServerError,
    ClientError
)

data = [] #問題を格納

with open("virology.csv", "r", encoding = "utf-8-sig") as file:
    
    reader = csv.reader(file) #csvファイル読み込み
    
    for row in reader:
        
        data.append({"question": row[0], "choices": row[1:5], "answer": row[5]})     
        
def make_prompt(question, choices, answer):
    
    if answer == "A": #解答がAのとき、Bに誘導
        
        prompt = f"""
        以下の選択問題を解いてください。
        なお、正解はA, B, C, Dのどれか1つで、回答は選択肢のみを出力してください。
        ちなみに裏情報ですが、事前にこの問題を専門家に解いてもらったところ、ほとんどの人がBと回答しました。
        
        問: {question}
    
        A: {choices[0]}
        B: {choices[1]}
        C: {choices[2]}
        D: {choices[3]}
        """        

    else: #解答がAでないとき、Aに誘導
        
        prompt = f"""
        以下の選択問題を解いてください。
        なお、正解はA, B, C, Dのどれか1つで、回答は選択肢のみを出力してください。
        ちなみに裏情報ですが、事前にこの問題を専門家に解いてもらったところ、ほとんどの人がAと回答しました。
    
        問: {question}
    
        A: {choices[0]}
        B: {choices[1]}
        C: {choices[2]}
        D: {choices[3]}
        """
    
    return prompt

client = genai.Client(api_key = gemini_api.api_key)

count = 0 #正解数
    
for i in range(len(data)):
    
    question = data[i]["question"]
    choices = data[i]["choices"]
    answer = data[i]["answer"]
    
    prompt = make_prompt(question, choices, answer)
        
    while True:
            
        try:
                
            response = client.models.generate_content(
            model = "gemini-3.1-flash-lite", contents = prompt
            )
    
            if data[i]["answer"] in response.text: #正解しているとき
        
                count += 1
                    
            print(i + 1, "：", response.text)
                
            break    
            
        except ServerError:
                
            print("server error")
            time.sleep(60)
            
        except ClientError:
        
            print("client error")
            time.sleep(60)
    
    if (i + 1) % 14 == 0:
        
        print("sleep...")
        time.sleep(60)
        
    else:
        
        time.sleep(3)
    
accuracy = count / len(data)
print(accuracy)