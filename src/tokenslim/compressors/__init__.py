"""M1 content-type compressors.

Each compressor is a configurable callable with the
``(text, content_type) -> str`` signature used by :mod:`tokenslim.router`.
Build them from a :class:`~tokenslim.config.Config` and register them in a
:class:`~tokenslim.router.ContentRouter`.
"""

from __future__ import annotations

from .logs import LogCompressor
from .search import SearchCompressor
from .smartcrusher import SmartCrusher

__all__ = ["SmartCrusher", "LogCompressor", "SearchCompressor"]
