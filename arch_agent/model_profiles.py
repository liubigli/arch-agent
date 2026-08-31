"""Per-model-family default parameters for ChatOllama.

Ollama's runtime context window and "thinking" (chain-of-thought) support
are not auto-negotiated by langchain_ollama: `num_ctx` silently falls back
to Ollama's own runtime default (historically 2048-4096 tokens) regardless
of a model's native context length, and `think` is only meaningful for
model families whose Modelfile actually implements it. For a tool-calling
agent that pipes large tool outputs back into the conversation, an
under-provisioned num_ctx is the easy way to end up with an AIMessage that
has no content (the model runs out of context/budget mid-thought before
writing the answer) without raising any exception the harness can catch.

This module gives each known model family a safer default; models we don't
recognize fall back to a conservative-but-larger-than-Ollama's-default profile.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    num_ctx: int
    think: bool | None = None
    num_predict: int | None = None


_DEFAULT_PROFILE = ModelProfile(num_ctx=8192)

# Keyed by the model name matched against the tag before ':' (so
# "qwen3.5:latest" is matched as "qwen3.5"), using startswith. Only set
# `think` for families known to support Ollama's thinking mode - forcing
# `think=True` on a model whose Modelfile doesn't implement it is untested
# and left as None (unset) instead.
_PROFILES: dict[str, ModelProfile] = {
    "qwen3.5": ModelProfile(num_ctx=32768, think=True, num_predict=4096),
    "qwen3": ModelProfile(num_ctx=32768, think=True, num_predict=4096),
    "qwen2.5": ModelProfile(num_ctx=16384),
    "qwen2": ModelProfile(num_ctx=16384),
    "deepseek-r1": ModelProfile(num_ctx=32768, num_predict=4096),
    "gemma4": ModelProfile(num_ctx=32768, think=True, num_predict=4096),
    "gpt-oss": ModelProfile(num_ctx=16384, think=True, num_predict=4096),
    "llama3.3": ModelProfile(num_ctx=16384),
    "llama3.2": ModelProfile(num_ctx=16384),
    "llama3.1": ModelProfile(num_ctx=16384),
    "llama3": ModelProfile(num_ctx=8192),
    "mixtral": ModelProfile(num_ctx=16384),
    "mistral-small3.2": ModelProfile(num_ctx=32768),
    "mistral-small": ModelProfile(num_ctx=32768),
    "mistral": ModelProfile(num_ctx=8192),
    "gemma2": ModelProfile(num_ctx=8192),
    "phi4": ModelProfile(num_ctx=16384),
    "phi3": ModelProfile(num_ctx=8192),
    "command-r": ModelProfile(num_ctx=16384),
}


def resolve_model_profile(model: str) -> ModelProfile:
    """Look up the default profile for an Ollama model tag.

    Matches by longest known prefix against both the full model tag and the
    family name before ':' (e.g. "qwen3:30b" and "qwen3.5:latest"), so
    version/size suffixes like ":latest" or ":8b" don't need their own entry.
    """
    full_name = model.lower()
    family_name = full_name.split(":", 1)[0]
    matches = [
        prefix
        for prefix in _PROFILES
        if full_name.startswith(prefix) or family_name.startswith(prefix)
    ]
    if not matches:
        return _DEFAULT_PROFILE
    best = max(matches, key=len)
    return _PROFILES[best]
