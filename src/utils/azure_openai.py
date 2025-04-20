import os
import requests

def azure_chat_completion(deployment: str,
                          messages: list[dict],
                          max_tokens: int = None,
                          temperature: float = None,
                          functions: list[dict] = None,
                          function_call: str = None) -> dict:
    """
    Send a chat completion request to Azure OpenAI using REST API.
    Args:
        deployment: Azure deployment name for the model.
        messages: List of message dicts for chat.
        max_tokens: Maximum tokens for completion.
        temperature: Sampling temperature.
        functions: List of function dicts for chat.
        function_call: Function call for chat.
    Returns:
        Parsed JSON response.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    version = os.getenv("AZURE_OPENAI_API_VERSION")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    headers = {"Content-Type": "application/json", "api-key": key}
    payload = {"messages": messages}
    if functions is not None:
        payload["functions"] = functions
    if function_call is not None:
        payload["function_call"] = function_call
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()
