const express = require('express');
const os = require('os');
const fs = require('fs');
const path = require('path');
const dgram = require('dgram');
const config = require('./config');
const photosRouter = require('./routes/photos');
const usersRouter = require('./routes/users');
const { router: aiRouter, triggerBatch } = require('./routes/ai');

const app = express();
app.use(express.json({ limit: '50mb' }));

// Track last activity — trigger AI batch after 5 min idle
let lastActivity = Date.now();
app.use((_req, _res, next) => { lastActivity = Date.now(); next(); });
setInterval(() => {
  if (Date.now() - lastActivity > 5 * 60_000) triggerBatch();
}, 60_000);

app.use('/photos', photosRouter);
app.use('/users', usersRouter);
app.use('/ai', aiRouter);

app.get('/status', (_req, res) => {
  const stat = fs.statfsSync(config.photoDir);
  const total_bytes = stat.bsize * stat.blocks;
  const available_bytes = stat.bsize * stat.bavail;
  res.json({
    name: 'Viviti Local',
    total_bytes,
    available_bytes,
    uptime: process.uptime(),
  });
});

app.get('/health', (_req, res) => res.json({ ok: true, viviti: true }));

// ── APK download ─────────────────────────────────────────────────────────────
const APK_PATH = path.join(__dirname, 'public', 'viviti.apk');

app.get('/app/download', (_req, res) => {
  if (!fs.existsSync(APK_PATH)) {
    return res.status(404).send('APK not yet available. Build with EAS and place at public/viviti.apk');
  }
  res.download(APK_PATH, 'viviti.apk');
});

// ── Landing page ──────────────────────────────────────────────────────────────
app.get('/', (_req, res) => {
  res.send(`<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Viviti</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #fefcfe; padding: 40px 24px; max-width: 400px; margin: 0 auto; }
    h1 { font-size: 34px; font-weight: 800; color: #257af0; margin-bottom: 4px; }
    .sub { color: #6b6070; font-size: 15px; margin-bottom: 36px; }
    label { font-size: 13px; font-weight: 600; color: #1a1118; display: block; margin-bottom: 6px; }
    input { width: 100%; border: 1.5px solid #e0dbe2; border-radius: 10px; padding: 14px 12px; font-size: 16px; margin-bottom: 8px; background: #fff; outline: none; -webkit-appearance: none; }
    input:focus { border-color: #257af0; }
    .hint { font-size: 12px; color: #9e96a4; margin-bottom: 24px; }
    button { width: 100%; background: #257af0; color: #fff; border: none; border-radius: 10px; padding: 16px; font-size: 16px; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.55; }
    .err { color: #f43f5e; font-size: 14px; margin-bottom: 16px; display: none; }
    #done { display: none; text-align: center; }
    #done h2 { font-size: 22px; font-weight: 700; color: #1a1118; margin-bottom: 8px; }
    #done .name { color: #257af0; }
    #done .desc { color: #6b6070; font-size: 15px; margin-bottom: 28px; }
    .dl-btn { display: block; background: #1a1118; color: #fff; border-radius: 10px; padding: 16px; font-size: 16px; font-weight: 700; text-decoration: none; text-align: center; margin-bottom: 16px; }
    .ios { font-size: 13px; color: #9e96a4; line-height: 1.6; margin-bottom: 24px; }
    .tip { font-size: 13px; color: #6b6070; background: #f5f3f7; border-radius: 10px; padding: 14px 16px; line-height: 1.6; }
    .tip strong { color: #1a1118; }
  </style>
</head>
<body>
  <div id="setup">
    <h1>Viviti</h1>
    <p class="sub">Your personal photo storage</p>
    <label for="nameInput">Your name</label>
    <input id="nameInput" type="text" placeholder="e.g. Prasad" autocomplete="off" autocorrect="off">
    <p class="hint">A folder with your name will be created on this device.</p>
    <div class="err" id="err"></div>
    <button id="btn" onclick="go()">Create Profile &amp; Download</button>
  </div>
  <div id="done">
    <h1>Viviti</h1><br>
    <h2>Welcome, <span class="name" id="doneName"></span>!</h2>
    <p class="desc">Your photo folder is ready on this device.</p>
    <a class="dl-btn" href="/app/download">⬇&nbsp; Download App (Android)</a>
    <p class="ios">iPhone? Search <strong>Viviti Local</strong> on the App Store.</p>
    <div class="tip">
      When setting up the app, enter your name exactly as:<br>
      <strong id="nameHint"></strong>
    </div>
  </div>
  <script>
    async function go() {
      const name = document.getElementById('nameInput').value.trim();
      const err = document.getElementById('err');
      err.style.display = 'none';
      if (!name) { err.textContent = 'Enter your name.'; err.style.display = 'block'; return; }
      document.getElementById('btn').disabled = true;
      try {
        const res = await fetch('/users/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        const data = await res.json();
        if (!res.ok) { err.textContent = data.error || 'Something went wrong.'; err.style.display = 'block'; document.getElementById('btn').disabled = false; return; }
        document.getElementById('doneName').textContent = data.name;
        document.getElementById('nameHint').textContent = data.name;
        document.getElementById('setup').style.display = 'none';
        document.getElementById('done').style.display = 'block';
      } catch (e) {
        err.textContent = 'Cannot reach server. Are you on the same WiFi?';
        err.style.display = 'block';
        document.getElementById('btn').disabled = false;
      }
    }
    document.getElementById('nameInput').addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
  </script>
</body>
</html>`);
});

function getLocalIp() {
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const iface of ifaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) return iface.address;
    }
  }
  return '127.0.0.1';
}

app.listen(config.port, () => {
  fs.mkdirSync(config.photoDir, { recursive: true });
  console.log(`Viviti local server running on port ${config.port}`);
  console.log(`Photo dir: ${config.photoDir}`);
  console.log(`Access at: http://${getLocalIp()}:${config.port}`);
});

// ── UDP discovery ─────────────────────────────────────────────────────────────
// Phones broadcast "viviti-discover" on UDP 55356; we reply with our IP+port.
const udpServer = dgram.createSocket('udp4');
udpServer.on('message', (msg, rinfo) => {
  if (msg.toString().trim() !== 'viviti-discover') return;
  const reply = Buffer.from(JSON.stringify({ viviti: true, ip: getLocalIp(), port: config.port }));
  udpServer.send(reply, rinfo.port, rinfo.address);
});
udpServer.bind(55356, () => {
  udpServer.setBroadcast(true);
  console.log('UDP discovery listening on port 55356');
});
