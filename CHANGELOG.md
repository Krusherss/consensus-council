# Changelog

## 0.3.0 — 2026-07-31

### Added

- Three-stage `deliberate()` / `adeliberate()` workflow: independent panel, rotated anonymous peer review, and chairman synthesis.
- Featured o3 + Gemini 2.5 Pro + Grok 4 panel with Claude Opus 4.6 as chairman; all models remain configurable through LiteLLM.
- Optional DuckDuckGo/Trafilatura search follow-up that asks the requesting model to revise with retrieved citations.
- Opt-in Markdown checkpoints and final audit artifacts.
- `deliberate` CLI command, CI test matrix, security policy, and dedicated deliberation tests.

### Changed

- Provider failures are recorded without echoing potentially sensitive exception details.
- Removed the obsolete Tavily helper and dependency; keyless DuckDuckGo sourcing is the supported search path.
- Removed unsupported production-use claims from the README.

### Compatibility

- Existing `vote()`, `debate()`, and `decide()` APIs remain available.
- The root-level standalone script is retained for v0.2 compatibility; new integrations should use the packaged API.
