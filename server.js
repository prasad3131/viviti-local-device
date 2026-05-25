const express = require('express');
const os = require('os');
const fs = require('fs');
const config = require('./config');
const photosRouter = require('./routes/photos');

const app = express();
app.use(express.json({ limit: '50mb' }));

app.use('/photos', photosRouter);

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

app.get('/health', (_req, res) => res.json({ ok: true }));

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
