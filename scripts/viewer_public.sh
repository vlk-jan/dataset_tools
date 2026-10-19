#!/bin/bash
# Browse the public HELHEST release: start the viewer on its URL root and open
# a browser at the index. Scenes are mirrored into the cache as they are viewed.
#
#   scripts/viewer_public.sh [viewer flags...]     # e.g. --port 8090 --no-notes
#
# Uses the checkout's .venv-viewer (or .venv) when there is one, else
# helhest-viewer from PATH. VIEWER_URL points it at another root.
set -euo pipefail

URL=${VIEWER_URL:-http://subtdata.felk.cvut.cz/HELHEST/}
REPO=$(cd "$(dirname "$0")/.." && pwd)

VIEWER=$(command -v helhest-viewer || true)
for venv in .venv-viewer .venv; do
    if [ -x "$REPO/$venv/bin/helhest-viewer" ]; then
        VIEWER=$REPO/$venv/bin/helhest-viewer
        break
    fi
done
: "${VIEWER:?helhest-viewer not found; install the viewer extra: pip install -e \"$REPO[viewer]\"}"

exec "$VIEWER" "$URL" --open "$@"
