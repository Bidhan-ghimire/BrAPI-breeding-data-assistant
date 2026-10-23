"""Console output a run can always write (Luna 2026-10-23, the T4 pass): on Windows a redirected stdout is cp1252, and a model's
U+2212 minus in the answer failed 6 runs at accept after a right analysis. A character the console's code page cannot hold is
replaced, never raised. Only the console transcript: answer.md, answer.json and the chat keep every character."""
from __future__ import annotations

import sys
from typing import Any


def say(text: str, **print_args: Any) -> None:
    """The console message, in the console's own code page."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), **print_args)
