from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse
from transformers import AutoModelForQuestionAnswering,AutoTokenizer, BertJapaneseTokenizer
import torch

# model_name = 'KoichiYasuoka/bert-base-japanese-wikipedia-ud-head'
model_name = 'local-model'

# 環境変数を設定してオフライモードにする
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def home(request):
    return render(request, 'home.html')

def reply(question):

     tokenizer = AutoTokenizer.from_pretrained(model_name)

     context = "私の名前は山田です。趣味は動画鑑賞とショッピングです。年齢は30歳です。出身は大阪府です。仕事は医者です。"
     inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
     input_ids = inputs["input_ids"].tolist()[0]
     output = model(**inputs)
     answer_start = torch.argmax(output.start_logits)  
     answer_end = torch.argmax(output.end_logits) + 1 
     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
     answer = answer.replace(' ', '')

     return answer


def bot_response(request):
     """
     HTMLフォームから受信したデータを返す処理
     http://127.0.0.1:8000/bot_response/として表示する
     """

     input_data = request.POST.get('input_text')
     if not input_data:
         return HttpResponse('<h2>空のデータを受け取りました。</h2>', status=400)

     bot_response = reply(input_data)
     http_response = HttpResponse()
     http_response.write(f"BOT: {bot_response}")

     return http_response