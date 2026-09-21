"""spike_ai - Behavior Cloning AI cho The Spike Cross Remaster."""

import sys

__version__ = "0.1.0"

# Console Windows mặc định cp1252 không in được tiếng Việt -> chuyển sang UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")
