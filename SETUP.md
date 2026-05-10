# Setup

## Base install (markitdown only, what most clients need)

```bash
pip3 install --break-system-packages -r requirements.txt
pip3 install --break-system-packages 'markitdown[pdf,docx,pptx,xlsx]'
```

The `[pdf,docx,pptx,xlsx]` extras add the format-specific dependency packages that markitdown needs (pdfplumber, mammoth, python-pptx, openpyxl). Skip them only if you're parsing HTML/CSV/JSON exclusively.

## Optional escalation backends

### Docling (table-heavy + scanned PDFs)

```bash
pip3 install --break-system-packages docling
```

First parse downloads the layout + table-detection model weights (~500 MB). Subsequent parses reuse the cache. No env vars needed.

### LlamaParse (cloud, BYOK)

```bash
pip3 install --break-system-packages llama-cloud-services
```

Set the API key in the MCP server's env block:

```json
"parse": {
  "env": {
    "LLAMA_CLOUD_API_KEY": "llx-..."
  }
}
```

Without the key, the LlamaParse backend reports `not available` and the router skips it. With the key, it becomes the deepest fallback for visually-complex PDFs.

## interpret tool: Anthropic API key

`interpret(source, instruction)` pipes parsed markdown into Claude. Requires `ANTHROPIC_API_KEY` in the MCP server's env:

```json
"parse": {
  "env": {
    "ANTHROPIC_API_KEY": "sk-ant-..."
  }
}
```

Without the key, `interpret` returns an error and the parse step still runs (so the markdown is available; only the LLM step fails).

## Registration

Already added to `[VAULT_ROOT]/.mcp.json` under the `parse` server name. Restart Claude Code to load. Validate with `claude mcp list`.

## Verify

```bash
cd ~/.claude/parse-mcp && python3 -c "import server; print('OK')"
python3 -c "import json; json.load(open('/path/to/vault/.mcp.json')); print('OK')"
```

If both print OK, restart Claude Code. The `parse` server should appear in `claude mcp list`.

## Troubleshooting

- `markitdown not installed`: rerun the base install above.
- `<backend> not available` in `list_backends()`: backend's deps or env vars are missing. Install per the section above.
- `input exceeds PARSE_MCP_MAX_BYTES`: bump the env var on the server registration. Default is 26214400 (25 MB).
- Empty markdown but no error: format-specific markitdown extra is missing (e.g. parsed a PDF without `[pdf]`). Reinstall with the relevant extras.
