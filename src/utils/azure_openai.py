import os
import requests
import logging

logger = logging.getLogger(__name__)

def azure_chat_completion(
    deployment: str,
    messages: list[dict],
    max_tokens: int = None,
    temperature: float = None,
    tools: list = None,           # 🆕 최신 OpenAI tools 스펙
    tool_choice: str | dict = None,  # 🆕 최신 OpenAI tools 스펙
    functions: list = None,      # ← 구버전 호환
    function_call: str | dict = None # ← 구버전 호환
) -> dict:
    """
    Send a chat completion request to Azure OpenAI using REST API.
    Args:
        deployment: Azure deployment name for the model.
        messages: List of message dicts for chat.
        max_tokens: Maximum tokens for completion.
        temperature: Sampling temperature.
        tools: List of tool schemas (for function calling, 최신)
        tool_choice: "auto" or {"name": ...} (최신)
        functions: List of function schemas (for function calling, 구버전)
        function_call: "auto" or {"name": ...} (구버전)
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
    if tools is not None:
        payload["tools"] = tools
    elif functions is not None:
        payload["functions"] = functions
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    elif function_call is not None:
        payload["function_call"] = function_call
    logger.debug(f"[azure_chat_completion] 요청 payload: {payload}")
    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error("Azure 요청 에러: %s\n응답 바디: %s", err, response.text)
        raise
    return response.json()

