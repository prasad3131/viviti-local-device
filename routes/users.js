const express = require('express');
const fs = require('fs');
const path = require('path');
const config = require('../config');

const router = express.Router();

router.post('/register', (req, res) => {
  const raw = String(req.body.name || '').trim();
  const name = path.basename(raw);
  if (!name) return res.status(400).json({ error: 'name required' });
  const dir = path.join(config.photoDir, name);
  if (!dir.startsWith(config.photoDir)) return res.status(400).json({ error: 'invalid name' });
  fs.mkdirSync(dir, { recursive: true });
  res.json({ ok: true, name });
});

router.get('/:name', (req, res) => {
  const name = path.basename(String(req.params.name || ''));
  const dir = path.join(config.photoDir, name);
  if (!dir.startsWith(config.photoDir) || !fs.existsSync(dir)) {
    return res.status(404).json({ error: 'User not found' });
  }
  res.json({ ok: true, name });
});

module.exports = router;
