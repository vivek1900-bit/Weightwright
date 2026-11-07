"""Safe parser for index strings like "[0:10, :]" (no eval)."""
from __future__ import annotations

import re

_PART = re.compile(r"^\s*(-?\d+)?\s*(?::\s*(-?\d+)?\s*)?$")


def parse_index(text: str | None, ndim: int) -> tuple:
    """Turn "[:, 0:128]" into a tuple of ints/slices. None or "" means everything."""
    if text is None or not str(text).strip():
        return tuple(slice(None) for _ in range(ndim))
    body = str(text).strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    parts = [p for p in body.split(",")]
    if len(parts) > ndim:
        raise ValueError(f"index {text!r} has {len(parts)} parts but tensor has {ndim} dims")
    out: list = []
    for p in parts:
        if ":" not in p:
            if not re.fullmatch(r"\s*-?\d+\s*", p):
                raise ValueError(f"bad index part {p!r}")
            out.append(int(p))
            continue
        m = _PART.match(p)
        if not m:
            raise ValueError(f"bad slice part {p!r}")
        # split manually to keep start/stop
        a, b = p.split(":")
        start = int(a) if a.strip() else None
        stop = int(b) if b.strip() else None
        out.append(slice(start, stop))
    while len(out) < ndim:
        out.append(slice(None))
    return tuple(out)
