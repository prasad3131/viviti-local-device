#!/bin/bash
# Viviti WiFi boot check
# Runs at startup — if the Pi has no active WiFi client connection, starts an AP hotspot
# so new users can configure their network from the Viviti app.

MODE_FILE="/opt/viviti/.wifi-mode"

log() { echo "[viviti-wifi] $*"; logger -t viviti-wifi "$*"; }

# Detect WiFi interface name (wlan0 on most Orange Pi / Raspberry Pi)
IFACE=$(nmcli -t -f DEVICE,TYPE device 2>/dev/null | grep ":wifi" | head -1 | cut -d: -f1)
IFACE=${IFACE:-wlan0}

log "Waiting for NetworkManager on $IFACE..."
sleep 12

# Check for an active client connection (non-AP IP, not in the 10.42.0.x hotspot range)
IP=$(ip -4 addr show "$IFACE" 2>/dev/null \
     | grep -oP '(?<=inet )\d+\.\d+\.\d+\.\d+' \
     | grep -v '^10\.42\.' \
     | head -1)

if [ -n "$IP" ]; then
  log "WiFi client active at $IP"
  echo "client" > "$MODE_FILE"
  exit 0
fi

# No client connection found — start AP hotspot
log "No WiFi client connection. Starting Viviti-Setup hotspot on $IFACE..."

nmcli connection delete viviti-hotspot 2>/dev/null || true

nmcli device wifi hotspot \
  ifname "$IFACE" \
  ssid "Viviti-Setup" \
  password "viviti123" \
  con-name viviti-hotspot

if [ $? -eq 0 ]; then
  echo "ap" > "$MODE_FILE"
  log "Hotspot started: SSID=Viviti-Setup  IP=10.42.0.1"
else
  echo "client" > "$MODE_FILE"
  log "Failed to start hotspot — defaulting to client mode"
fi
