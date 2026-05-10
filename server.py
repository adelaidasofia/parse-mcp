"""parse-mcp: best-of-the-best document parsing MCP for Adelaida + Mycelium clients.

ONE tool surface, multiple deterministic backends behind it (markitdown
default, Docling for tables/scanned, LlamaParse for visually-complex
BYOK), plus a Claude-powered interpretation layer for "summarize / extract
X" workflows. The router picks the right backend per format and falls
back automatically when one returns empty or errors.

Tools:
- parse: file-path-or-URL to markdown, with full audit trail
- parse_url: shortcut for HTTP(S) inputs
- interpret: parse, then ask Claude over the parsed markdown
- list_backends: which backends are installed and ready
- benchmark: run every available backend on the same input, compare

Architecture follows MCP Build Runbook v1: FastMCP v3.2.3+, stdio
transport, registered in `[VAULT_ROOT]/.mcp.json`. No network listeners,
no new daemons. All work runs in-process.
"""
from __future__ import annotations

import datetime
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Ensure the bundled `backends/` package is importable when FastMCP
# launches us via `python3 server.py` from elsewhere.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fastmcp import FastMCP

import router  # noqa: E402
from interpret import interpret as _interpret  # noqa: E402

# 25 MB default ceiling. Override via env. Keeps the synchronous parse
# path responsive and matches the memory-runtime-pro parse-layer ceiling
# so audit log shapes stay aligned across surfaces.
_MAX_BYTES_DEFAULT = 25 * 1024 * 1024

mcp = FastMCP("parse-mcp")


def _max_bytes() -> int:
    raw = os.environ.get("PARSE_MCP_MAX_BYTES")
    if not raw:
        return _MAX_BYTES_DEFAULT
    try:
        v = int(raw)
    except ValueError:
        return _MAX_BYTES_DEFAULT
    return v if v > 0 else _MAX_BYTES_DEFAULT


def _read_path(path: str) -> tuple[bytes | None, str, str | None]:
    """Read a filesystem path. Returns (data, filename, error)."""
    p = Path(path).expanduser()
    if not p.exists():
        return None, p.name or path, f"not found: {p}"
    if not p.is_file():
        return None, p.name or path, f"not a file: {p}"
    try:
        size = p.stat().st_size
    except OSError as exc:
        return None, p.name or path, str(exc)
    if size > _max_bytes():
        return None, p.name, f"input exceeds PARSE_MCP_MAX_BYTES ({_max_bytes()} bytes)"
    try:
        return p.read_bytes(), p.name, None
    except OSError as exc:
        return None, p.name, str(exc)


def _read_url(url: str) -> tuple[bytes | None, str, str | None]:
    """Fetch an HTTP(S) URL. Returns (data, filename, error)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return None, url, "url must use http or https"
    filename = url.rsplit("/", 1)[-1] or "download"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "parse-mcp/1.0 (Mycelium AI)",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read(_max_bytes() + 1)
            if len(data) > _max_bytes():
                return None, filename, f"response exceeds PARSE_MCP_MAX_BYTES ({_max_bytes()} bytes)"
            return data, filename, None
    except urllib.error.HTTPError as exc:
        return None, filename, f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return None, filename, f"URLError: {exc.reason}"
    except Exception as exc:
        return None, filename, f"{type(exc).__name__}: {exc}"


@mcp.tool()
def parse(
    source: str,
    backend: str | None = None,
    hints: dict | None = None,
) -> dict:
    """Convert a document to markdown. Routes across backends automatically.

    Args:
        source: A filesystem path or http(s) URL.
        backend: Force a specific backend (e.g. "markitdown", "docling",
            "llamaparse"). Omit to let the router pick + fall back.
        hints: Optional backend hints (reserved for future use).

    Returns a dict with `markdown`, the winning `backend` name, the
    `format` token, latency in ms, byte count, and a `chain` of every
    backend attempted (success and failure) for the audit trail.
    """
    if source.startswith("http://") or source.startswith("https://"):
        data, filename, err = _read_url(source)
    else:
        data, filename, err = _read_path(source)
    if err is not None or data is None:
        return {
            "markdown": "",
            "backend": backend or "router",
            "format": "unknown",
            "bytes_in": 0,
            "latency_ms": 0,
            "error": err,
            "chain": [],
            "chosen_strategy": "default",
        }
    result = router.route(data, filename=filename, backend=backend, hints=hints)
    return result.to_dict()


@mcp.tool()
def parse_url(url: str, backend: str | None = None) -> dict:
    """Shortcut for parsing an HTTP(S) URL. Same return shape as `parse`."""
    return parse(url, backend=backend)


@mcp.tool()
def parse_to_vault(
    source: str,
    vault_folder: str | None = None,
    backend: str | None = None,
    overwrite: bool = True,
) -> dict:
    """Parse a document and write it to the vault as a markdown note.

    Replaces the standalone `⚙️ Meta/scripts/markitdown_to_vault.py`
    shell script: same output shape, but with the router's full audit
    trail (which backend ran, fallback chain, latency) baked into the
    frontmatter.

    Args:
        source: filesystem path or http(s) URL.
        vault_folder: target folder for the note. Default:
            `<VAULT_ROOT>/📥 Inbox/Converted/`. The folder is created
            if missing.
        backend: force a specific backend (default: router pick).
        overwrite: when False, refuse to overwrite an existing note;
            return error in that case. Default True (idempotent).

    Returns `{path, backend, format, bytes_in, latency_ms, chain[],
    error}`. `path` is the absolute path of the written note, or empty
    string on error.
    """
    parsed = parse(source, backend=backend)
    if parsed.get("error") or not parsed.get("markdown", "").strip():
        return {
            "path": "",
            "backend": parsed.get("backend"),
            "format": parsed.get("format"),
            "bytes_in": parsed.get("bytes_in", 0),
            "latency_ms": parsed.get("latency_ms", 0),
            "error": parsed.get("error") or "parse produced empty markdown",
            "chain": parsed.get("chain", []),
        }

    vault_root = Path(
        os.environ.get(
            "VAULT_ROOT",
            str(Path.home() / "Desktop" / "Adelaida Notes"),
        )
    )
    if vault_folder:
        out_dir = Path(vault_folder).expanduser()
        if not out_dir.is_absolute():
            out_dir = vault_root / out_dir
    else:
        out_dir = vault_root / "📥 Inbox" / "Converted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stem: from filename for paths, from URL last segment for URLs.
    if source.startswith("http://") or source.startswith("https://"):
        stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "download"
    else:
        stem = Path(source).stem
    dest = out_dir / f"{stem}.md"

    if dest.exists() and not overwrite:
        return {
            "path": "",
            "backend": parsed.get("backend"),
            "format": parsed.get("format"),
            "bytes_in": parsed.get("bytes_in", 0),
            "latency_ms": parsed.get("latency_ms", 0),
            "error": f"destination exists and overwrite=False: {dest}",
            "chain": parsed.get("chain", []),
        }

    today = datetime.date.today().isoformat()
    frontmatter = (
        f"---\n"
        f"creationDate: {today}\n"
        f"type: converted\n"
        f"source: {source}\n"
        f"source_format: {parsed.get('format', 'unknown')}\n"
        f"converted_by: parse-mcp\n"
        f"backend: {parsed.get('backend', 'router')}\n"
        f"latency_ms: {parsed.get('latency_ms', 0)}\n"
        f"bytes_in: {parsed.get('bytes_in', 0)}\n"
        f"---\n\n"
    )
    try:
        dest.write_text(frontmatter + parsed["markdown"], encoding="utf-8")
    except OSError as exc:
        return {
            "path": "",
            "backend": parsed.get("backend"),
            "format": parsed.get("format"),
            "bytes_in": parsed.get("bytes_in", 0),
            "latency_ms": parsed.get("latency_ms", 0),
            "error": f"write failed: {exc}",
            "chain": parsed.get("chain", []),
        }

    return {
        "path": str(dest),
        "backend": parsed.get("backend"),
        "format": parsed.get("format"),
        "bytes_in": parsed.get("bytes_in", 0),
        "latency_ms": parsed.get("latency_ms", 0),
        "error": None,
        "chain": parsed.get("chain", []),
    }


@mcp.tool()
def interpret(
    source: str,
    instruction: str,
    backend: str | None = None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
) -> dict:
    """Parse the document, then ask Claude over the parsed markdown.

    Use this for "summarize this PDF", "extract every action item",
    "what does this contract say about termination" style requests.
    The router parses first (cheap, deterministic), then Claude reads
    the markdown and answers. Cache hits across calls in the same
    session reuse the parsed text for free input tokens.

    Args:
        source: filesystem path or http(s) URL.
        instruction: what you want Claude to do with the document.
        backend: force a specific parser backend (default: router pick).
        model: which Claude model. Sonnet 4.6 is default; bump to Opus
            for hard reasoning, drop to Haiku for speed/cost.
        max_tokens: response cap.

    Returns the Claude answer + the parse audit trail + token usage.
    Requires ANTHROPIC_API_KEY in the MCP server env.
    """
    if source.startswith("http://") or source.startswith("https://"):
        data, filename, err = _read_url(source)
    else:
        data, filename, err = _read_path(source)
    if err is not None or data is None:
        return {
            "answer": "",
            "model": model,
            "instruction": instruction,
            "truncated": False,
            "error": err,
            "parse": {
                "markdown": "",
                "backend": backend or "router",
                "format": "unknown",
                "bytes_in": 0,
                "latency_ms": 0,
                "error": err,
                "metadata": {},
                "chosen_strategy": "default",
                "chain": [],
            },
            "metadata": {},
        }
    result = _interpret(
        data,
        filename=filename,
        instruction=instruction,
        backend=backend,
        model=model,
        max_tokens=max_tokens,
    )
    return result.to_dict()


@mcp.tool()
def list_backends() -> dict:
    """Report which parse backends are installed + which are missing.

    Returns a list of `{name, available, module}` entries. Use this to
    debug "why did the router fall back to markitdown" by checking
    whether docling/llamaparse are actually available.
    """
    return {"backends": router.list_backends(), "max_bytes": _max_bytes()}


@mcp.tool()
def benchmark(source: str) -> dict:
    """Run every available backend on the same input, compare results.

    Diagnostic tool. Returns one ParseResult per available backend with
    latency, byte counts, error state, and metadata so you can see at
    a glance which parser handles a given document best.
    """
    if source.startswith("http://") or source.startswith("https://"):
        data, filename, err = _read_url(source)
    else:
        data, filename, err = _read_path(source)
    if err is not None or data is None:
        return {"error": err, "results": []}
    results = router.benchmark(data, filename=filename)
    return {"results": [r.to_dict() for r in results], "filename": filename}


if __name__ == "__main__":
    mcp.run()
