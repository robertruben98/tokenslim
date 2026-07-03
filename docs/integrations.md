# Integrations

All integrations share one philosophy: **duck typing, zero SDK imports**.
TokenSlim never imports `openai`, `anthropic`, `litellm`, `langchain`, `agno`
or `strands` in its core paths — wrappers patch by attribute shape, fail
silently back to the original behavior, and the optional extras only exist to
pin the frameworks for *your* environment.

## OpenAI & Anthropic clients

`with_tokenslim(client)` patches the client in place (and returns it) so every
request's `messages` array is compressed before it leaves the process. Both
sync and async clients work:

```python
from openai import OpenAI
from tokenslim import with_tokenslim

client = with_tokenslim(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=messages)
```

```python
from anthropic import Anthropic
from tokenslim import with_tokenslim

client = with_tokenslim(Anthropic())
client.messages.create(model="claude-sonnet-4-5", max_tokens=1024, messages=messages)
```

Pass a `Config` to control the compression:
`with_tokenslim(client, config=Config(min_bytes=500))`. When
`Config.output_reduction` (env: `TOKENSLIM_OUTPUT_REDUCTION`) is `"balanced"`
or `"aggressive"`, the wrapper also appends output-brevity instructions after
compression — to the system message for OpenAI-shaped clients, to the
top-level `system` kwarg for Anthropic.

!!! note
    Pass `messages` as a keyword argument — positional message arrays bypass
    the wrapper.

## LiteLLM

`TokenSlimLiteLLMCallback` implements LiteLLM's proxy callback shape
(`pre_call_hook` / `async_pre_call_hook`) and compresses the messages of every
call routed through it:

```python
from tokenslim import TokenSlimLiteLLMCallback

callback = TokenSlimLiteLLMCallback()
# register per LiteLLM's docs, e.g. in the proxy: callbacks: [callback]
```

## LangChain

Three entry points, all duck-typed on `.type`/`.content` (messages) and
`.page_content` (documents) — the `langchain` extra is optional:

```python
from tokenslim import wrap_chat_model, compress_messages, compress_documents

wrap_chat_model(chat_model)        # invoke/ainvoke/stream now compress inputs

trimmed_history = compress_messages(memory_messages)     # new objects, inputs untouched

docs = compress_documents(retriever_docs, query="user question")
```

- `wrap_chat_model(model, config=None)` patches `invoke`/`ainvoke`/`stream` on
  the instance; wrapping is idempotent, non-message inputs (plain strings,
  PromptValues) pass through untouched.
- `compress_documents(docs, config=None, query=None)` feeds `query` into the
  BM25 relevance scorer so rows matching the user's question survive
  aggressive crushing — ideal between the retriever and the prompt.

## Agno

Two hooks, use either or both:

```python
from agno.agent import Agent
from tokenslim import tokenslim_agno_tool_hook, wrap_agno_model

# 1. Compress every tool's output as it is produced.
agent = Agent(tools=[...], tool_hooks=[tokenslim_agno_tool_hook])

# 2. Or compress the full message payload right before each model call.
wrap_agno_model(agent.model)
```

`wrap_agno_model` duck-patches whichever of `invoke` / `ainvoke` / `response`
/ `aresponse` exist; it is idempotent and never raises.

## Strands

`TokenSlimStrandsHooks` is a hook provider — the agent's conversation is
compressed in place before every model invocation, including strands' typeless
`{"text": ...}` / `{"toolResult": ...}` content blocks:

```python
from strands import Agent
from tokenslim import TokenSlimStrandsHooks

agent = Agent(model=..., hooks=[TokenSlimStrandsHooks()])
```

`strands` is imported lazily at registration only; both the
`BeforeModelCallEvent` (≥1.8) and `BeforeModelInvocationEvent` (earlier 1.x)
event names are supported.

## Any framework: compress_tool_output

The lowest common denominator — a bulletproof `str -> str` helper you can call
from any hook in any framework:

```python
from tokenslim import compress_tool_output

def my_tool_middleware(output: str) -> str:
    return compress_tool_output(output)   # never raises; returns input on any problem
```

## Reverse proxy

`tokenslim proxy` runs a stdlib-only OpenAI-compatible reverse proxy: POST
bodies to `/v1/chat/completions` get their `messages` compressed before being
forwarded upstream; everything else (including `/v1/responses`) passes through
untouched. SSE streaming is relayed chunk-by-chunk, headers (including
`Authorization`) are preserved, and any parse/compression error falls back to
forwarding the original bytes.

```bash
tokenslim proxy --port 8787 --upstream https://api.openai.com
```

Point your client at it — no code changes beyond the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1")   # api_key from env as usual
```

`GET /health` answers locally; one line per request is logged to stderr with
the path, `orig -> new` token counts and the savings ratio. Configuration is
env-based: `TOKENSLIM_PROXY_PORT` (default `8787`), `TOKENSLIM_UPSTREAM`
(default `https://api.openai.com`), plus every other `TOKENSLIM_*` knob.

The proxy is also available programmatically: `run_proxy(port=None,
upstream=None)` and `make_proxy_server(...)` are exported from `tokenslim`.

## Docker

The published `Dockerfile` packages the proxy as a container with a
`/health`-based `HEALTHCHECK` and a non-root user:

```bash
docker build -t tokenslim .
docker run --rm -p 8787:8787 -e TOKENSLIM_UPSTREAM=https://my-gateway.example tokenslim
```

Full instructions: [TokenSlim in Docker](DOCKER.md).

## Environment-variable escape hatch

Anything that shells out can be wrapped without touching its code:

```bash
tokenslim wrap -- my-agent-command --its-flags
```

runs the command with `TOKENSLIM_ENABLED=true` injected into its environment —
useful when the child process itself embeds TokenSlim.
