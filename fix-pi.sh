#!/bin/bash
# Run once on Pi to fix WiFi reliability permanently
# Usage: cd /opt/viviti && git pull && bash fix-pi.sh
set -e

echo "=== Disabling WiFi power management ==="
mkdir -p /etc/NetworkManager/conf.d
cat > /etc/NetworkManager/conf.d/99-wifi-pm.conf << 'EOF'
[connection]
wifi.powersave = 2
EOF

# udev rule: disables power save at the driver level every time WiFi starts
# Needed for Unisoc chips that ignore NM's power save setting
cat > /etc/udev/rules.d/70-wifi-pm.rules << 'EOF'
ACTION=="add", SUBSYSTEM=="net", KERNEL=="wlan*", RUN+="/usr/sbin/iw dev %k set power_save off"
EOF
udevadm control --reload-rules

echo "=== Setting WiFi auto-reconnect on all saved networks ==="
for CON in $(nmcli -t -f NAME,TYPE con show | grep ':wifi$' | cut -d: -f1 | grep -v viviti-hotspot); do
  echo "  autoconnect: $CON"
  nmcli con modify "$CON" connection.autoconnect yes connection.autoconnect-priority 50
done

echo "=== Hardening viviti.service ==="
cat > /etc/systemd/system/viviti.service << 'EOF'
[Unit]
Description=Viviti local device server
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/viviti
ExecStart=/usr/bin/node server.js
Restart=always
RestartSec=5
StartLimitIntervalSec=0
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "=== Applying changes ==="
systemctl daemon-reload
systemctl restart NetworkManager
sleep 3
systemctl restart viviti

echo ""
echo "=== Verification ==="
iw dev wlan0 get power_save 2>/dev/null || echo "(power_save check skipped)"
ip addr show wlan0 | grep "inet "
systemctl is-active viviti
echo "Done."
