# TokenSlim in Docker

The image runs `tokenslim proxy` — an OpenAI-compatible reverse proxy that
compresses the `messages` array of `POST /v1/chat/completions` bodies before
forwarding them to the upstream API. Everything else (including
`/v1/responses`) passes through untouched, and SSE streaming responses are
relayed chunk-by-chunk.

## Build

```bash
docker build -t tokenslim .
```

## Run

```bash
# Proxy to the default upstream (https://api.openai.com)
docker run --rm -p 8787:8787 tokenslim

# Proxy to a custom OpenAI-compatible upstream
docker run --rm -p 8787:8787 -e TOKENSLIM_UPSTREAM=https://my-llm-gateway.example tokenslim
```

Point your client at the proxy instead of the API — the `Authorization`
header is forwarded as-is:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1")  # api_key from env as usual
```

Check it is alive:

```bash
curl http://localhost:8787/health
# {"status": "ok", "version": "..."}
```

## Configuration

All TokenSlim settings are plain `TOKENSLIM_*` environment variables
(`docker run -e ...`). The most relevant ones:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOKENSLIM_UPSTREAM` | `https://api.openai.com` | Upstream base URL requests are forwarded to. |
| `TOKENSLIM_PROXY_PORT` | `8787` | Port the proxy listens on inside the container. |
| `TOKENSLIM_TARGET_RATIO` | `0.2` | Target compression ratio. |
| `TOKENSLIM_ENABLED` | `true` | Set `false` to pass everything through uncompressed. |
| `TOKENSLIM_TELEMETRY` | on | Set `off` to disable anonymous telemetry. |

Run `tokenslim doctor` for the full list:

```bash
docker run --rm tokenslim doctor
```

The image's entrypoint is the `tokenslim` CLI itself, so any subcommand works
(`docker run --rm tokenslim --help`).

## Notes

- The container runs as a non-root `tokenslim` user.
- One log line per request is written to stderr with the path, original vs
  compressed token counts, and the savings ratio (`docker logs -f <container>`).
- A `HEALTHCHECK` hitting `GET /health` is built in.
