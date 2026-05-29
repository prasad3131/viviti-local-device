#!/bin/bash
# Viviti OTA update — runs via systemd timer every 4 hours
set -e
cd /opt/viviti

BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
git fetch origin main --quiet 2>&1
REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "")

if [ "$BEFORE" = "$REMOTE" ] || [ -z "$REMOTE" ]; then
  echo "$(date -Iseconds) viviti-update: already up to date ($BEFORE)"
  exit 0
fi

echo "$(date -Iseconds) viviti-update: updating $BEFORE -> $REMOTE"
git pull origin main --quiet 2>&1
npm install --production --quiet 2>&1
echo "$(date -Iseconds) viviti-update: restarting viviti service"
systemctl restart viviti
echo "$(date -Iseconds) viviti-update: done"
