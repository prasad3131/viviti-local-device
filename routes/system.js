const express = require('express');
const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');

const router = express.Router();
const MODE_FILE = path.join(__dirname, '..', '.wifi-mode');

function readMode() {
  try { return fs.readFileSync(MODE_FILE, 'utf8').trim(); } catch { return 'client'; }
}

function writeMode(m) {
  try { fs.writeFileSync(MODE_FILE, m); } catch {}
}

function getWifiIface(cb) {
  execFile('sh', ['-c', "nmcli -t -f DEVICE,TYPE device | grep ':wifi' | head -1 | cut -d: -f1"],
    (_err, stdout) => cb((stdout || '').trim() || 'wlan0'));
}

function startHotspot(iface) {
  execFile('nmcli', ['connection', 'delete', 'viviti-hotspot'], () => {
    execFile('nmcli', [
      'device', 'wifi', 'hotspot',
      'ifname', iface,
      'ssid', 'Viviti-Setup',
      'password', 'viviti123',
      'con-name', 'viviti-hotspot',
    ], (err) => { if (!err) writeMode('ap'); });
  });
}

// GET /system/wifi/status
router.get('/wifi/status', (_req, res) => {
  const mode = readMode();
  execFile('nmcli', ['-t', '-f', 'IP4.ADDRESS', 'dev', 'show', 'wlan0'], (_err, stdout) => {
    const line = (stdout || '').trim().split('\n')[0] || '';
    const ip = line.split(':')[1]?.split('/')[0] || null;
    res.json({ mode, ip });
  });
});

// GET /system/wifi/networks
// ?rescan=true triggers a live scan (~5s); omit for cached results (instant)
router.get('/wifi/networks', (req, res) => {
  const args = ['-t', '-f', 'SSID,SIGNAL,SECURITY', 'device', 'wifi', 'list'];
  if (req.query.rescan === 'true') args.push('--rescan', 'yes');

  execFile('nmcli', args, { timeout: 15000 }, (err, stdout) => {
    if (err) return res.status(500).json({ error: 'Scan failed' });

    const seen = new Set();
    const networks = [];

    for (const line of (stdout || '').trim().split('\n').filter(Boolean)) {
      // terse format: SSID:SIGNAL:SECURITY — split from right to handle colons in SSID
      const last = line.lastIndexOf(':');
      const prev = line.lastIndexOf(':', last - 1);
      if (last < 0 || prev < 0) continue;
      const ssid = line.slice(0, prev).replace(/\\:/g, ':').trim();
      const signal = parseInt(line.slice(prev + 1, last), 10);
      const security = line.slice(last + 1).trim();
      if (!ssid || ssid === '--' || seen.has(ssid)) continue;
      seen.add(ssid);
      networks.push({ ssid, signal, security: security || 'Open' });
    }

    networks.sort((a, b) => b.signal - a.signal);
    res.json(networks);
  });
});

// POST /system/wifi/connect  { ssid, password }
// Responds immediately (202), then transitions the Pi to the new network.
router.post('/wifi/connect', (req, res) => {
  const { ssid, password } = req.body || {};
  if (!ssid) return res.status(400).json({ error: 'ssid required' });

  // Respond before we tear down the current network so the client gets the ack.
  res.status(202).json({ ok: true, transitioning: true });

  const wasAp = readMode() === 'ap';

  setTimeout(() => {
    getWifiIface(iface => {
      // Stop hotspot if running
      execFile('nmcli', ['connection', 'down', 'viviti-hotspot'], () => {
        const args = ['--wait', '25', 'device', 'wifi', 'connect', ssid];
        if (password) args.push('password', password);

        execFile('nmcli', args, { timeout: 30000 }, (err) => {
          if (err) {
            // Connection failed — restart hotspot if we were in AP mode so user can retry
            if (wasAp) startHotspot(iface);
          } else {
            writeMode('client');
          }
        });
      });
    });
  }, 300); // small delay to let the HTTP response flush
});

module.exports = { router, readMode };
