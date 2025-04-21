# test_openai.py
from dotenv import load_dotenv
import os, openai

# 1) .env 로드
load_dotenv()

# 2) 환경변수 확인 (옵션)
print("KEY   :", os.getenv("AZURE_OPENAI_API_KEY")[:8] + "…")
print("ENDPT :", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("VER   :", os.getenv("AZURE_OPENAI_API_VERSION"))
print("DEPLOY:", os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4"))

# 3) SDK 설정
openai.api_type    = "azure"
openai.api_base    = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key     = os.getenv("AZURE_OPENAI_API_KEY")

# 4) 짧게 ping 요청
client = openai.OpenAI(api_key=openai.api_key)
resp = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4"),
    messages=[
        {"role":"system", "content":"You are a test assistant."},
        {"role":"user",   "content":"Ping"}
    ],
    max_tokens=1
)
print("✅ 응답:", resp.choices[0].message.content)