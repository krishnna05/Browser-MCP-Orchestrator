#!/bin/bash
# start_mcp.sh
# Browser-MCP Orchestrator — Playwright MCP startup wrapper for Streamlit Cloud (Linux).
#
# Root cause of the cloud error:
#   '@playwright/mcp' bundles its own npm playwright in the npx cache.
#   That playwright looks for Chromium in a path set by PLAYWRIGHT_BROWSERS_PATH.
#   By default it differs from Python playwright's install path.
#   This script forces both to share ONE common browser cache and installs
#   Chromium via the npm playwright that @playwright/mcp will actually use.

set -euo pipefail

# ── Shared browser cache ──────────────────────────────────────────────────────
# Both npm playwright (used by @playwright/mcp) and Python playwright will read
# browsers from this directory once PLAYWRIGHT_BROWSERS_PATH is exported.
export PLAYWRIGHT_BROWSERS_PATH="${HOME}/.cache/ms-playwright"
mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}"

# ── Install Chromium via npm playwright ───────────────────────────────────────
# npx playwright install resolves to the playwright version that npx has cached
# for @playwright/mcp, ensuring the browser binary lives exactly where the MCP
# server expects it.
echo "[start_mcp.sh] Installing Chromium via npm playwright..."
npx --yes playwright install chromium 2>&1 || true

# ── Also install system deps (no-op if already present) ──────────────────────
npx --yes playwright install-deps chromium 2>&1 || true

echo "[start_mcp.sh] Launching @playwright/mcp..."
exec npx --yes @playwright/mcp@latest \
    --browser chromium \
    --headless \
    --no-sandbox \
    "$@"
