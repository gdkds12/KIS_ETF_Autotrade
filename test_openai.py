# test_openai.py
from dotenv import load_dotenv
import os
from src.utils.azure_openai import azure_chat_completion

# 1) .env 로드
load_dotenv()

# 2) 환경변수 확인 (옵션)
print("KEY   :", os.getenv("AZURE_OPENAI_API_KEY")[:8] + "…")
print("ENDPT :", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("VER   :", os.getenv("AZURE_OPENAI_API_VERSION"))
print("DEPLOY:", os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4"))

# 4) 짧게 ping 요청
resp_json = azure_chat_completion(
    deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4"),
    messages=[
        {"role":"system","content":"You are a test assistant."},
        {"role":"user","content":"Ping"}
    ],
    max_tokens=1
)
print("✅ 응답:", resp_json["choices"][0]["message"]["content"])