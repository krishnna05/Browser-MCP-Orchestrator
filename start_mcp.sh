#!/bin/bash
# Browser-MCP Orchestrator — Playwright MCP startup wrapper for Streamlit Cloud (Linux).

set -euo pipefail

export PLAYWRIGHT_BROWSERS_PATH="${HOME}/.cache/ms-playwright"
mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}"

echo "[start_mcp.sh] Installing Chrome for Testing..."
npx --yes @playwright/mcp@latest install-browser chrome-for-testing

echo "[start_mcp.sh] Launching @playwright/mcp..."
exec npx --yes @playwright/mcp@latest \
    --browser chrome \
    --headless \
    --no-sandbox \
    "$@"
