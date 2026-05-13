# parse-mcp

One MCP, many parsers. Default markitdown (free, fast, MIT). Escalate to Docling (table-heavy, scanned PDFs) or LlamaParse (cloud, BYOK) when markitdown's quality isn't enough. Plus an `interpret` tool that pipes parsed markdown into Claude for "summarize / extract X" so you stop juggling parsers and anthropic skills.

## Install

Open Claude Code, paste:

    /plugin marketplace add adelaidasofia/parse-mcp
    /plugin install parse-mcp@parse-mcp

<details><summary>Legacy install</summary>

Manual install (pre-plugin-marketplace). See `SETUP.md` for full details.

```bash
pip3 install --break-system-packages -r requirements.txt
pip3 install --break-system-packages 'markitdown[pdf,docx,pptx,xlsx]'
```

Then register the server in your client's `.mcp.json`:

```json
{
  "mcpServers": {
    "parse": {
      "command": "python3",
      "args": ["/absolute/path/to/parse-mcp/server.py"]
    }
  }
}
```

</details>

## Tools

| Tool | What it does |
|---|---|
| `parse(source, backend?, hints?)` | File path or http(s) URL to markdown. Router picks backend, falls back on empty/error. Returns markdown plus a chain of every backend attempted. |
| `parse_url(url, backend?)` | Shortcut for HTTP(S) inputs. Same return shape as `parse`. |
| `parse_to_vault(source, vault_folder?, backend?, overwrite?)` | Parse + write the result as a markdown note in the vault. Default folder: `<VAULT_ROOT>/📥 Inbox/Converted/`. Frontmatter records source, format, backend, latency, bytes_in. Replaces the standalone `markitdown_to_vault.py` shell script. |
| `interpret(source, instruction, backend?, model?, max_tokens?)` | Parse first, then ask Claude over the parsed markdown. Cache hits reuse parsed text for free input tokens. |
| `list_backends()` | Which backends are installed + which are missing. Diagnostic. |
| `benchmark(source)` | Run every available backend on the same input. Compare latency + output side by side. |

## Backends (priority order)

1. **markitdown** (default, MIT, base install). PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPub, ZIP. Fast, deterministic.
2. **docling** (optional, `pip install docling`). Best for complex tables (97.9% on benchmark) + scanned PDFs. Downloads model weights on first run.
3. **llamaparse** (optional, BYOK, `pip install llama-cloud-services` + `LLAMA_CLOUD_API_KEY`). Cloud, cleanest output on visually-complex PDFs.

## Routing strategy

- `parse(source)` with no `backend` arg: router picks based on file format, falls back if backend errors or returns empty.
- `parse(source, backend="docling")`: force a specific backend, no fallback. Diagnostic mode.
- Unavailable backends are skipped (logged in the chain), never errored.

## Architecture

FastMCP v3.2.3+, stdio transport, Python 3.13+. Registered in `[VAULT_ROOT]/.mcp.json`. No daemons, no listeners, no model weights downloaded by default.

See `SETUP.md` for install + per-backend opt-in.

## Related MCPs

Same author, same architecture pattern (FastMCP, draft+confirm on writes, vault auto-export where applicable):

- [slack-mcp](https://github.com/adelaidasofia/slack-mcp) — multi-workspace Slack
- [imessage-mcp](https://github.com/adelaidasofia/imessage-mcp) — macOS iMessage
- [whatsapp-mcp](https://github.com/adelaidasofia/whatsapp-mcp) — WhatsApp via whatsmeow
- [apollo-mcp](https://github.com/adelaidasofia/apollo-mcp) — Apollo.io CRM + sequences
- [google-workspace-mcp](https://github.com/adelaidasofia/google-workspace-mcp) — Gmail / Calendar / Drive / Docs / Sheets
- [substack-mcp](https://github.com/adelaidasofia/substack-mcp) — Substack writing + analytics


## Telemetry

This plugin sends a single anonymous install signal to `myceliumai.co` the first time it loads in a Claude Code session on a given machine.

**What is sent:**
- Plugin name (e.g. `slack-mcp`)
- Plugin version (e.g. `0.1.0`)

**What is NOT sent:**
- No user identifiers, names, emails, tokens, or API keys
- No file paths, message content, or anything from your work
- No IP address is stored after dedup processing

**Why:** Helps the maintainer know which plugins people actually install, so attention goes to the ones that get used.

**Opt out:** Set the environment variable `MYCELIUM_NO_PING=1` before launching Claude Code. The hook will skip the network call entirely. Already-pinged installs leave a sentinel at `~/.mycelium/onboarded-<plugin>` — delete it if you want to reset state.

## License

MIT. See `LICENSE`.

---

Built by Adelaida Diaz-Roa. Full install or team version at [diazroa.com](https://diazroa.com).
