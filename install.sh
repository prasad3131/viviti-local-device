#!/bin/bash
set -e

echo "=== Viviti Local — Setup ==="

# ── WiFi watchdog ────────────────────────────────────────────────────────────
cat > /usr/local/bin/wifi-watchdog.sh << 'EOF'
#!/bin/bash
while true; do
  if ! nmcli -t -f STATE g | grep -q connected; then
    nmcli con up shivaru-2g
  fi
  sleep 30
done
EOF
chmod +x /usr/local/bin/wifi-watchdog.sh

cat > /etc/systemd/system/wifi-watchdog.service << 'EOF'
[Unit]
Description=WiFi Watchdog
After=network.target

[Service]
ExecStart=/usr/local/bin/wifi-watchdog.sh
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# ── Viviti server service ────────────────────────────────────────────────────
cat > /etc/systemd/system/viviti.service << 'EOF'
[Unit]
Description=Viviti Local Server
After=network.target

[Service]
ExecStart=/usr/bin/node /home/viviti-local-device/server.js
WorkingDirectory=/home/viviti-local-device
Restart=always
RestartSec=3
Environment=PHOTO_DIR=/home/orangepi/photos

[Install]
WantedBy=multi-user.target
EOF

# ── Enable and start ─────────────────────────────────────────────────────────
mkdir -p /home/orangepi/photos
systemctl daemon-reload
systemctl enable wifi-watchdog viviti
systemctl restart wifi-watchdog viviti

echo ""
echo "=== Done! ==="
echo "Server: $(systemctl is-active viviti)"
echo "WiFi watchdog: $(systemctl is-active wifi-watchdog)"
echo "Access at: http://$(hostname -I | awk '{print $1}'):3000"
