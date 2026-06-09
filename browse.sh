#!/usr/bin/env bash
set -euo pipefail

url="${1:-about:blank}"

# -----------------------------
# Detect video URLs → mpv
# -----------------------------
if [[ "$url" =~ ^https?://((www|m)\.)?(youtube\.com/(watch|shorts)|youtu\.be/|instagram\.com/(reel|p)/|facebook\.com/.*/videos/|fb\.watch/|tiktok\.com/.*/video/) ]]; then
    mpv --no-terminal --player-operation-mode=pseudo-gui "$url" &
else
    /usr/local/bin/firebox "$url" 
fi
exit 0
