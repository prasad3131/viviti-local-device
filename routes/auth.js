const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const KEY_FILE = path.join(__dirname, '..', '.device-key');

// Generate once on first boot, persist forever (survives git pulls and restarts)
function getOrCreateKey() {
  if (fs.existsSync(KEY_FILE)) {
    return fs.readFileSync(KEY_FILE, 'utf8').trim();
  }
  const key = crypto.randomBytes(4).toString('hex').toUpperCase()
    + '-'
    + crypto.randomBytes(4).toString('hex').toUpperCase();
  fs.writeFileSync(KEY_FILE, key, { mode: 0o600 });
  return key;
}

const DEVICE_KEY = getOrCreateKey();

// Middleware: validates X-Viviti-Key header on protected routes
function requireKey(req, res, next) {
  const provided = (req.headers['x-viviti-key'] || '').toUpperCase();
  if (provided !== DEVICE_KEY) {
    return res.status(401).json({ error: 'Invalid or missing device key' });
  }
  next();
}

module.exports = { DEVICE_KEY, requireKey };
