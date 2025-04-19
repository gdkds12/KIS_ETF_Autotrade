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