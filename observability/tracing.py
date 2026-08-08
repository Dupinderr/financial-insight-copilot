"""Langfuse tracing for the multi-agent graph.

Every agent node runs inside a span, and every LLM call is recorded as a
generation with model, latency and token usage. All of it degrades to a no-op
when Langfuse credentials are absent, so the graph runs identically untraced.

Set these in .env to turn tracing on:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com     # or https://us.cloud.langfuse.com
"""

import os
import time
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

_client = None
_initialised = False
_init_error = None


class _NullSpan:
    """Stand-in span used when tracing is off, so call sites stay uniform."""

    def update(self, **kwargs):
        pass

    def update_trace(self, **kwargs):
        pass


def get_client():
    """Return an authenticated Langfuse client, or None if unavailable."""
    global _client, _initialised, _init_error

    if _initialised:
        return _client

    _initialised = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        _init_error = "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"
        return None

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        if not client.auth_check():
            _init_error = "Langfuse auth_check() failed — check the keys and host region"
            return None

        _client = client
    except Exception as e:
        _init_error = f"Langfuse init failed: {e}"
        return None

    return _client


def tracing_status() -> dict:
    client = get_client()
    return {
        "enabled": client is not None,
        "reason": _init_error,
        "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") if client else None,
    }


@contextmanager
def trace_span(name: str, as_type: str = "span", input=None, metadata=None):
    """Wrap a unit of work in a Langfuse observation.

    Yields the span (or a null object). Callers set results with
    `span.update(output=...)`.
    """
    client = get_client()

    if client is None:
        yield _NullSpan()
        return

    with client.start_as_current_observation(
        name=name, as_type=as_type, input=input, metadata=metadata
    ) as span:
        try:
            yield span
        except Exception as e:
            span.update(level="ERROR", status_message=str(e))
            raise


@contextmanager
def trace_run(name: str, input=None, metadata=None):
    """Root span for one full graph run. Yields (span, trace_url)."""
    client = get_client()

    if client is None:
        yield _NullSpan(), None
        return

    with client.start_as_current_observation(
        name=name, as_type="chain", input=input, metadata=metadata
    ) as span:
        url = None
        try:
            url = client.get_trace_url()
        except Exception:
            pass
        try:
            yield span, url
        except Exception as e:
            span.update(level="ERROR", status_message=str(e))
            raise
        finally:
            client.flush()


MAX_ATTEMPTS = 4


def _complete(groq_client, model: str, prompt: str, max_tokens: int, temperature: float):
    """Groq call with backoff on rate limits.

    The eval harness fires ~90 calls in a burst, which is enough to trip the
    free tier's per-minute limit; without this the run dies partway through.
    """
    delay = 2.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            return groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            message = str(e).lower()

            # A per-day quota will not clear within any backoff we could sit
            # through, so fail immediately with an actionable message rather
            # than burning four attempts on it.
            if "per day" in message or "tpd" in message or "requests per day" in message:
                raise RuntimeError(
                    "Groq daily token quota exhausted (free tier is 100k tokens/day). "
                    "The quota resets on a rolling daily window — retry later, or "
                    "switch GROQ model/tier. Original error: " + str(e)
                ) from e

            transient = "rate" in message or "429" in message or "503" in message
            if not transient or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= 2


def llm_call(
    prompt: str,
    *,
    name: str,
    max_tokens: int = 700,
    temperature: float = 0.0,
    model: str | None = None,
) -> str:
    """Single traced Groq call — the one place every agent's LLM traffic goes.

    Reuses the Groq client and model from data_agent so there is one API key
    path and one model choice across v1 and v2.
    """
    from data_agent import MODEL, client as groq_client

    model = model or MODEL
    span_client = get_client()
    started = time.time()

    if span_client is None:
        response = _complete(groq_client, model, prompt, max_tokens, temperature)
        return response.choices[0].message.content.strip()

    with span_client.start_as_current_observation(
        name=name,
        as_type="generation",
        input=prompt,
        model=model,
        model_parameters={"max_tokens": max_tokens, "temperature": temperature},
    ) as gen:
        response = _complete(groq_client, model, prompt, max_tokens, temperature)
        text = response.choices[0].message.content.strip()

        usage = getattr(response, "usage", None)
        if usage is not None:
            gen.update(
                usage_details={
                    "input": getattr(usage, "prompt_tokens", 0) or 0,
                    "output": getattr(usage, "completion_tokens", 0) or 0,
                    "total": getattr(usage, "total_tokens", 0) or 0,
                }
            )

        gen.update(output=text, metadata={"latency_s": round(time.time() - started, 3)})
        return text


def flush():
    client = get_client()
    if client is not None:
        client.flush()
