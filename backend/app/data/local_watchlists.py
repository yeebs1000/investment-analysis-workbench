"""App-local watchlists — broker-independent named lists of symbol codes.

IBKR's TWS/Gateway socket API (the one this app connects through via ib_async)
does not expose watchlists at all — they live only in IBKR's separate Client
Portal Web API. So IBKR-only users had no broker-side watchlist to read. These
lists are stored by the app itself as a JSON map ``{group_name: [codes]}`` and
merged into the same watchlist pipeline Moomoo groups use, so the feature works
identically regardless of which broker(s) are linked.

Plain read/write, no broker dependency. Writes are temp-file-then-rename (like
`app/analytics/performance.py`) under a module lock so a concurrent read never
sees a half-written file.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

_LOCK = threading.Lock()


def _store_path() -> Path:
    d = Path(__file__).resolve().parents[2] / "data_store"
    d.mkdir(parents=True, exist_ok=True)
    return d / "watchlists.json"


def _load() -> dict[str, list[str]]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Normalize to {str: [str]} so a hand-edited/corrupt file can't crash a read.
    return {str(k): [str(c) for c in v] for k, v in data.items() if isinstance(v, list)}


def _save(data: dict[str, list[str]]) -> None:
    path = _store_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def groups() -> list[str]:
    return list(_load().keys())


def codes(group: str) -> list[str]:
    return _load().get(group.strip(), [])


def has_group(group: str) -> bool:
    return group.strip() in _load()


def add(group: str, code: str) -> None:
    """Add a code to a group, creating the group if new. De-duplicated."""
    g, c = group.strip(), code.strip().upper()
    if not g or not c:
        raise ValueError("Group name and code are required.")
    with _LOCK:
        data = _load()
        lst = data.setdefault(g, [])
        if c not in lst:
            lst.append(c)
        _save(data)


def remove(group: str, code: str) -> None:
    g, c = group.strip(), code.strip().upper()
    with _LOCK:
        data = _load()
        if g in data:
            data[g] = [x for x in data[g] if x != c]
            _save(data)


def delete(group: str) -> None:
    g = group.strip()
    with _LOCK:
        data = _load()
        if g in data:
            del data[g]
            _save(data)


def demo() -> None:
    """Self-check: exercise the round-trip against a temp store."""
    import tempfile

    global _store_path
    orig = _store_path
    tmpdir = tempfile.mkdtemp()
    _store_path = lambda: Path(tmpdir) / "watchlists.json"  # noqa: E731
    try:
        assert groups() == []
        add("Ideas", "us.aapl")           # lowercase -> upper, group created
        add("Ideas", "US.MSFT")
        add("Ideas", "US.AAPL")           # duplicate -> no-op
        assert has_group("Ideas")
        assert codes("Ideas") == ["US.AAPL", "US.MSFT"], codes("Ideas")
        remove("Ideas", "us.aapl")        # case-insensitive remove
        assert codes("Ideas") == ["US.MSFT"], codes("Ideas")
        add("Other", "SG.D05")
        assert set(groups()) == {"Ideas", "Other"}
        delete("Ideas")
        assert not has_group("Ideas")
        try:
            add("", "X")
            assert False, "empty group should raise"
        except ValueError:
            pass
        print("local_watchlists demo OK")
    finally:
        _store_path = orig


if __name__ == "__main__":
    demo()
