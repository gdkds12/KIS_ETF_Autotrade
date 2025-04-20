import os
import requests

def azure_chat_completion(deployment: str, messages: list[dict], max_tokens: int = None, temperature: float = None) -> dict:
    """
    Send a chat completion request to Azure OpenAI using REST API.
    Args:
        deployment: Azure deployment name for the model.
        messages: List of message dicts for chat.
        max_tokens: Maximum tokens for completion.
        temperature: Sampling temperature.
    Returns:
        Parsed JSON response.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("AZURE_OPENAI_API_VERSION")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    headers = {"Content-Type": "application/json", "api-key": key}
    payload = {"messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
