# Canonical CI gate. `make ci` is the ONE command both CI (test.yml, auto-managed
# by gh-harden-repos.sh) and the local pre-push gate (ci-test) run, so they cannot
# drift. Mirrors the test job: pytest tests/ -v, tolerating exit 5 (no tests
# collected) exactly as the auto-managed workflow does.
.PHONY: ci
ci:
	pytest tests/ -v || [ $$? -eq 5 ]

# Regenerate the parse-fidelity matrix from the fixture corpus. Run after any
# parse-backend version or config change — the committed matrix rots otherwise.
# markitdown is base; `pip install docling` and set LLAMA_CLOUD_API_KEY to
# populate those columns.
.PHONY: eval
eval:
	python tests/eval/score_parse_fidelity.py
