# Reversibility (CCR)

CCR — **Compressed-Content-Record** — is what makes TokenSlim's lossy
compression safe. Whenever a compressor drops material, it:

1. stores the dropped original in a content-addressed **CCR store**, and
2. leaves a small machine-readable **marker** in the output whose hash is the
   store key.

Nothing is silently deleted; the model (or a tool acting for it) can always ask
for the full data back.

## Markers

The canonical marker is a compact text token:

```text
<<ccr:HASH N reason>>
```

- `HASH` — content hash of the dropped material (the store key).
- `N` — how many items/lines were dropped.
- `reason` — a short tag such as `middle-elided`, `lines-elided`, `rows-elided`.

Text compressors emit it inside a greppable one-liner:

```text
[tokenslim:ccr] 212 lines-elided <<ccr:9f3ab61c2e77d105 212 lines-elided>>
```

For JSON arrays the marker is wrapped in a sentinel object so it survives
re-serialisation:

```json
{"_ccr_dropped": "<<ccr:9f3a… 492 middle-elided>>", "__tokenslim_ccr__": {"...": "detail"}}
```

Marker helpers are all public: `make_marker`, `parse_marker`, `find_markers`,
`strip_markers`, and the `CCRMarker` dataclass.

## Retrieving the originals

`compress()` exposes the store it used on the stats object — always retrieve
through it:

```python
from tokenslim import compress, retrieve, find_markers

out, stats = compress(messages)

for marker in find_markers(out[1]["content"]):
    original = retrieve(marker.hash, store=stats.store)
```

!!! warning "Pass `stats.store`"
    With the default in-memory backend, `retrieve(hash)` *without* an explicit
    store builds a fresh empty store and returns `None`. Either pass
    `store=stats.store`, or use a persistent backend (`sqlite` / `redis`) and
    pass the matching `config`.

## Store backends

The store is a small protocol — `put(original) -> hash`, `get(hash) ->
str | None` — with three implementations, dispatched by
`get_store(config)` on `Config.ccr_backend`:

| Backend | Class | Use case |
| --- | --- | --- |
| `memory` (default) | `InMemoryCCRStore` | Same-process retrieval; gone when the process exits |
| `sqlite` | `SQLiteCCRStore` | Persistent local file — retrieval across processes/restarts |
| `redis` | `RedisCCRStore` | Shared store for distributed agents (needs `redis` installed) |

| Knob | Default | Meaning |
| --- | --- | --- |
| `ccr` | `true` | Emit markers and stash dropped originals |
| `ccr_backend` | `memory` | `memory`, `sqlite`, or `redis` |
| `ccr_path` | `tokenslim_ccr.sqlite3` | SQLite database file (sqlite backend) |
| `ccr_ttl` | `None` | Seconds before a stored record expires (`None` = keep forever) |
| `redis_url` | `redis://localhost:6379/0` | Connection string (redis backend) |

```python
from tokenslim import Config, compress, retrieve

cfg = Config(ccr_backend="sqlite", ccr_path="/var/lib/agent/ccr.sqlite3")
out, stats = compress(messages, options=cfg)
# later, even in another process:
original = retrieve("9f3ab61c2e77d105", config=cfg)
```

With `ccr=False` compression still runs, but SmartCrusher drops silently and
no markers or store writes happen — use it only when reversibility genuinely
does not matter.

### Through the proxy

`tokenslim proxy` builds **one** shared store at start-up and reuses it for
every request, so the originals it drops outlive the request that created them.
Because the default `memory` backend cannot survive that long, the proxy
transparently upgrades it to a persistent `SQLiteCCRStore` at `ccr_path`
(honouring `ccr_ttl`); set `TOKENSLIM_CCR_BACKEND` to `sqlite`/`redis` to pick
your own. Expand a marker the proxy emitted with the local retrieve endpoint:

```bash
curl "http://localhost:8787/tokenslim/retrieve?hash=9f3ab61c2e77d105"
# -> {"hash": "9f3ab61c2e77d105", "original": "…"}
# ?marker=<<ccr:HASH N reason>> is accepted too.
```

or from the CLI, pointed at the same backend:

```bash
TOKENSLIM_CCR_BACKEND=sqlite TOKENSLIM_CCR_PATH=./tokenslim_ccr.sqlite3 \
  tokenslim retrieve "<<ccr:9f3ab61c2e77d105 492 middle-elided>>"
```

Both keep working after the request finishes and after the proxy restarts, as
long as the record is still within its TTL.

## Scoped retrieval with CCRContext

`CCRContext` guards a retrieval tool against fetching arbitrary store contents:
it only serves hashes for markers the model has actually *seen* in the
conversation.

```python
from tokenslim import CCRContext, compress

out, stats = compress(messages)
ctx = CCRContext(store=stats.store)     # share the store compression used
ctx.track(out)                          # registers every marker in the messages

ctx.retrieve("9f3ab61c2e77d105")        # only works for tracked hashes
```

Wire `ctx.retrieve` up as the implementation of a "expand compressed content"
tool and the model can drill into any elision on demand — the agent decides
what it needs, TokenSlim guarantees it can get it.

## Faithfulness guarantees

The bundled eval suite (`tokenslim evals`) enforces the CCR contract on every
fixture: compress → collect markers → fetch each original from the store →
reconstruct → **byte-compare** with what was dropped, and independently verify
that must-keep rows (errors) survived in the visible output. Any unfaithful
fixture fails the suite (exit code 1) and CI.
