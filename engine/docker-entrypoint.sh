#!/bin/sh
# A container's port is usually reachable from outside (a RunPod proxy URL is public), so the
# engine does not start without an API key unless that is asked for explicitly.
set -e
if [ -z "$TINYLM_API_KEY" ] && [ -z "$TINYLM_ALLOW_NO_API_KEY" ]; then
    echo "TINYLM_API_KEY is not set: anyone who can reach port $TINYLM_PORT could use the engine." >&2
    echo "Set TINYLM_API_KEY, or TINYLM_ALLOW_NO_API_KEY=1 on a private network." >&2
    exit 1
fi
mkdir -p "$TINYLM_DATA_DIR"
exec "$@"
