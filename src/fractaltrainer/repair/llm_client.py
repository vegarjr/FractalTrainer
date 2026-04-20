"""VENDORED from /home/vegar/Documents/Fractal/evolution/stage11_introspection/llm_client.py
Source commit: a910a9202c9d5e8b9f5f3c2c1c9e7d6b5a4b3c2a (Task 9/10)
Original author: Vegar Ratdal (Fractal project)
License: MIT (same ownership)

Modifications vs upstream:
  * Added make_local_llm_client() for a local llama.cpp server
    (OpenAI-compatible /v1/chat/completions). Zero dependencies — uses
    urllib.request. Default target is http://127.0.0.1:8080 (the Qwen2.5
    Coder-7B server set up via docs/local_llm_setup.md).
"""

import json
import os
import subprocess
import urllib.request
import urllib.error
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


def make_local_llm_client(
    base_url: str = "http://127.0.0.1:8080",
    model: str = "qwen25-coder-7b",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    timeout: float = 300.0,
    ping_on_create: bool = True,
):
    """Create a callable that sends prompts to a local llama.cpp server
    via OpenAI-compatible /v1/chat/completions.

    Target: a llama-server instance running on base_url (default
    http://127.0.0.1:8080) — see docs/local_llm_setup.md for the Qwen2.5
    Coder-7B setup on GTX 1050 / Vulkan that this defaults to.

    Zero Python dependencies — uses urllib.request. Returns the same
    `fn(system_prompt, user_prompt) -> response_text` signature as the
    CLI and API variants above, so the repair loop can swap backends by
    constructor only.

    Args:
        base_url: llama-server base, without /v1 suffix.
        model: label passed to the server; llama.cpp ignores this but
               OpenAI clients expect it present.
        max_tokens: hard cap on generated tokens per response.
        temperature: sampling temperature. 0.3 is a sensible default for
                     code-generation / repair tasks; raise to 0.7+ for
                     diverse-strategy generation (e.g. Snake-teacher use).
        timeout: per-request timeout seconds. 300s is generous for the
                 ~8 tok/s throughput on GTX 1050.
        ping_on_create: health-check the server at construction time and
                        raise RuntimeError with a setup hint if it's not
                        reachable. Pass False for tests with mock servers.

    Raises:
        RuntimeError: if ping_on_create=True and the /health endpoint
                      doesn't respond with status=ok.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    health_url = base_url.rstrip("/") + "/health"

    if ping_on_create:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("status") != "ok":
                    raise RuntimeError(
                        f"llama-server at {base_url} returned "
                        f"status={data.get('status')!r} "
                        "— expected 'ok'.")
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            raise RuntimeError(
                f"Cannot reach llama-server at {health_url}: {e}. "
                "Start it with:\n"
                "  export LD_LIBRARY_PATH=$HOME/.local/llama.cpp/llama-b8857:"
                "$LD_LIBRARY_PATH\n"
                "  ~/.local/llama.cpp/llama-b8857/llama-server \\\n"
                "    -m ~/.local/models/qwen25-coder-7b.gguf \\\n"
                "    --host 127.0.0.1 --port 8080 \\\n"
                "    --n-gpu-layers 20 --ctx-size 4096 &\n"
                "(see docs/local_llm_setup.md for full instructions)"
            ) from e

    def call(system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Local LLM request failed (HTTP {e.code}): "
                f"{e.read()[:500]!r}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"Local LLM connection error: {e}") from e

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Local LLM returned no choices: {payload!r}")
        msg = choices[0].get("message", {})
        text = msg.get("content")
        if not isinstance(text, str):
            raise RuntimeError(
                f"Local LLM response malformed: {payload!r}")
        return text

    return call
