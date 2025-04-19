"""Very small registry for LLM‑callable functions.

Usage:

    from src.utils.registry import command

    @command
    def get_balance():
        ...
"""

from __future__ import annotations

import inspect
from typing import Callable, Dict, Any

COMMANDS: Dict[str, Callable[..., Any]] = {}

# Global runtime object (lazy‑set later)
ORCHESTRATOR = None  # type: Any

# Utility so main.py can tell us about the live orchestrator
def set_orchestrator(obj: Any) -> None:
    """Store a reference to the (running) orchestrator so wrappers can reach it."""
    global ORCHESTRATOR
    ORCHESTRATOR = obj

def command(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator – registers *fn* so the LLM can discover & invoke it.

    It also attaches a minimal OpenAI function‑spec (``._oas``) which we later pass
    to :pyfunc:`openai.chat.completions.create`.
    """

    name = fn.__name__
    sig = inspect.signature(fn)

    # Very naïve JSON schema – all params are plain strings.
    parameters_schema = {
        "type": "object",
        "properties": {
            k: {"type": "string"} for k in sig.parameters.keys()
        },
        "required": list(sig.parameters.keys()),
    }

    COMMANDS[name] = fn
    fn._oas = {
        "name": name,
        "description": fn.__doc__ or "",
        "parameters": parameters_schema,
    }

    return fn

# --- NEW WRAPPERS --- 
@command
def get_balance() -> dict:
    """현재 계좌 예수금·총자산."""
    if ORCHESTRATOR is None:
        return {"error": "orchestrator not ready"}
    return ORCHESTRATOR.broker.get_balance()


@command
def get_positions() -> list:
    """보유 포지션."""
    if ORCHESTRATOR is None:
        return []
    return ORCHESTRATOR.broker.get_positions()


@command
def get_market_summary(query: str) -> str:
    """‘query’에 맞춰 InfoCrawler+LLM으로 시장 동향을 요약합니다."""
    if ORCHESTRATOR is None:
        return "(orchestrator not ready)"
    # Pass the query to the InfoCrawler method
    return ORCHESTRATOR.info_crawler.get_market_summary(query) 