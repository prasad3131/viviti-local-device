#!/bin/bash
# Run once on Pi to fix WiFi idle drops and set static IP
# Usage: cd /opt/viviti && git pull && bash fix-pi.sh
set -e

echo "=== Fixing DNS permanently ==="
# Stop NetworkManager from ever touching resolv.conf again
if ! grep -q "dns=none" /etc/NetworkManager/NetworkManager.conf; then
  sed -i '/^\[main\]/a dns=none' /etc/NetworkManager/NetworkManager.conf
fi
echo "nameserver 8.8.8.8" > /etc/resolv.conf
chattr +i /etc/resolv.conf  # make immutable so nothing can overwrite it

echo "=== Disabling WiFi power management ==="
mkdir -p /etc/NetworkManager/conf.d
cat > /etc/NetworkManager/conf.d/99-wifi-pm.conf << 'EOF'
[connection]
wifi.powersave = 2
EOF

echo "=== Setting static IP 192.168.1.213 ==="
CONNECTION=$(nmcli -t -f NAME con show --active | grep -v "^lo" | grep -v "^viviti" | head -1)
echo "Connection: $CONNECTION"
nmcli con modify "$CONNECTION" \
  ipv4.method manual \
  ipv4.addresses 192.168.1.213/24 \
  ipv4.gateway 192.168.1.1 \
  ipv4.dns "8.8.8.8 8.8.4.4"

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
iwconfig wlan0 | grep "Power Management"
ip addr show wlan0 | grep "inet "
systemctl is-active viviti
echo "Done. Pi will now hold 192.168.1.213 permanently and WiFi stays on."
