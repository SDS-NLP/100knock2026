#JMMLUのいずれかの科目を大規模言語モデルに解答させ、その正解率を求めよ。

import csv
from google import genai
import gemini_api
import time
from google.genai.errors import ServerError
from google.genai.errors import ClientError

data = [] #問題を格納

with open("virology.csv", "r", encoding = "utf-8-sig") as file:
    
    reader = csv.reader(file) #csvファイル読み込み
    
    for row in reader:
        
        data.append({"question": row[0], "choices": row[1:5], "answer": row[5]})
        
def make_prompt(question, choices): #dataのquestionとchoicesを埋め込んでプロンプトを作成
    
    prompt = f"""
    以下の選択問題を解いてください。
    なお、正解はA, B, C, Dのどれか1つで、回答は選択肢のみを出力してください。
    
    問: {question}
    
    A: {choices[0]}
    B: {choices[1]}
    C: {choices[2]}
    D: {choices[3]}
    """
    
    return prompt

count = 0 #正解数

client = genai.Client(api_key = gemini_api.api_key)

for i in range(len(data)):
    
    question = data[i]["question"]
    choices = data[i]["choices"]
    
    prompt = make_prompt(question, choices)
    
    while True:
        try:
            
            response = client.models.generate_content(
                model = "gemini-3.1-flash-lite", contents = prompt
            )
    
            if data[i]["answer"] in response.text: #正解しているとき
        
                count += 1
                
            print(i + 1, "：", response.text, count)    
            
            break
        
        except ServerError:
            
            print("server error")
            time.sleep(60)
        
        except ClientError:
            
            print("client error")
            time.sleep(60)      
                
    if (i + 1) % 14 == 0:
        
        print("sleep...")
        time.sleep(60) #60秒ストップ
    
    else:
        
        time.sleep(3) #一応、各問題間も3秒あける

accuracy = count / len(data) #正解率

print("正解率：", accuracy)

#正解率：0.5866666666666667