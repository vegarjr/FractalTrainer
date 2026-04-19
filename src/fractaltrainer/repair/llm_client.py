"""VENDORED from /home/vegar/Documents/Fractal/evolution/stage11_introspection/llm_client.py
Source commit: a910a9202c9d5e8b9f5f3c2c1c9e7d6b5a4b3c2a (Task 9/10)
Original author: Vegar Ratdal (Fractal project)
License: MIT (same ownership)
Modifications: none — both CLI and API backends are reusable unchanged.
"""

import os
import subprocess
from typing import Optional


def make_claude_cli_client(claude_bin: Optional[str] = None):
    """Create a callable that sends prompts via the Claude Code CLI.

    Uses `claude --print` which runs through the user's Max subscription
    at no additional API cost.

    Returns:
        fn(system_prompt, user_prompt) -> response_text
    """
    bin_path = claude_bin or "claude"

    try:
        subprocess.run(
            [bin_path, "--version"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            f"Claude CLI not found at '{bin_path}'. "
            "Install Claude Code or specify claude_bin=")

    def call(system_prompt: str, user_prompt: str) -> str:
        result = subprocess.run(
            [bin_path, "--print",
             "--system-prompt", system_prompt,
             "--model", "sonnet"],
            input=user_prompt,
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI failed (rc={result.returncode}): "
                f"{result.stderr[:500]}")
        return result.stdout

    return call


def make_claude_client(api_key: Optional[str] = None,
                       model: str = "claude-sonnet-4-6",
                       max_tokens: int = 4096):
    """Create a callable that sends prompts to Claude API.

    Returns:
        fn(system_prompt, user_prompt) -> response_text
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "No API key. Set ANTHROPIC_API_KEY or pass api_key=")

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "pip install anthropic — required for real-LLM repair loop")

    client = anthropic.Anthropic(api_key=key)

    def call(system_prompt: str, user_prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    return call
