# test_azure.py
import os, requests
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4")
version    = os.getenv("AZURE_OPENAI_API_VERSION")
key        = os.getenv("AZURE_OPENAI_API_KEY")

url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
headers = {
    "Content-Type": "application/json",
    "api-key": key
}
payload = {
    "messages": [
        {"role":"system","content":"You are a test assistant."},
        {"role":"user","content":"Ping"}
    ],
    "max_tokens": 1
}

resp = requests.post(url, headers=headers, json=payload)
print(resp.status_code, resp.json())