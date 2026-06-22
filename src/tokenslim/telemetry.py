"""Anonymous telemetry (opt-out).

What this collects: only **anonymous aggregate compression metrics** — content
type, original/compressed token counts, ratio, and the compressor name. Never
the payload, never the dropped originals, never anything that identifies a user
or machine.

Opt out with ``TOKENSLIM_TELEMETRY=off`` (or any falsey value), or
``Config(telemetry=False)``.

This is a **local, offline stub**: events are buffered in-process via the
default :class:`NullTelemetry` sink, which makes no network call. A real
exporter can subclass :class:`TelemetrySink` and be installed with
:func:`set_sink`; the shipped default sends nothing anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "TelemetryEvent",
    "TelemetrySink",
    "NullTelemetry",
    "set_sink",
    "get_sink",
    "record_event",
]


@dataclass(frozen=True)
class TelemetryEvent:
    """An anonymous compression event. Contains no payload content."""

    content_type: str
    orig_tokens: int
    new_tokens: int
    compressor: str

    @property
    def ratio(self) -> float:
        if self.orig_tokens == 0:
            return 0.0
        return 1.0 - (self.new_tokens / self.orig_tokens)


@runtime_checkable
class TelemetrySink(Protocol):
    """Receives anonymous telemetry events."""

    def emit(self, event: TelemetryEvent) -> None: ...


class NullTelemetry:
    """Default sink — buffers events in-process, sends nothing anywhere.

    Keeping the events locally lets tests assert on them and lets callers read
    back what *would* have been exported, without any network egress.
    """

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()


_sink: TelemetrySink = NullTelemetry()


def set_sink(sink: TelemetrySink) -> None:
    """Install the active telemetry sink (e.g. a real exporter)."""
    global _sink
    _sink = sink


def get_sink() -> TelemetrySink:
    """Return the active telemetry sink."""
    return _sink


def record_event(
    content_type: str,
    orig_tokens: int,
    new_tokens: int,
    compressor: str,
    *,
    enabled: bool = True,
) -> None:
    """Emit a telemetry event to the active sink, unless ``enabled`` is False."""
    if not enabled:
        return
    _sink.emit(TelemetryEvent(content_type, orig_tokens, new_tokens, compressor))
