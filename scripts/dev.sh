#!/usr/bin/env bash
# Start the desktop app in development mode. The app starts the engine itself.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

[[ -x "$(engine_python)" ]] || bash "$REPO_ROOT/scripts/setup-engine.sh"
cd "$REPO_ROOT/desktop"
[[ -d node_modules ]] || npm install
npm start
