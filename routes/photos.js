const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const config = require('../config');

const router = express.Router();

const IMAGE_RE = /\.(jpg|jpeg|png|gif|heic|raw|cr2|arw|nef|dng)$/i;

function safeDirPath(userPath) {
  if (!userPath) return config.photoDir;
  const parts = String(userPath).split('/').map(p => path.basename(p)).filter(Boolean);
  const full = path.join(config.photoDir, ...parts);
  if (!full.startsWith(config.photoDir)) return config.photoDir;
  return full;
}

router.get('/folders', (req, res) => {
  try {
    const dir = safeDirPath(req.query.path || '');
    fs.mkdirSync(dir, { recursive: true });
    const folders = fs.readdirSync(dir, { withFileTypes: true })
      .filter(e => e.isDirectory())
      .map(e => e.name);
    res.json({ folders });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.post('/folders', (req, res) => {
  const parentPath = req.body.path || '';
  const name = path.basename(String(req.body.name || ''));
  if (!name) return res.status(400).json({ error: 'name required' });
  const dir = path.join(safeDirPath(parentPath), name);
  if (!dir.startsWith(config.photoDir)) return res.status(400).json({ error: 'invalid path' });
  if (fs.existsSync(dir)) return res.status(409).json({ error: 'Folder already exists' });
  fs.mkdirSync(dir, { recursive: true });
  res.json({ ok: true, name });
});

router.get('/', (req, res) => {
  const offset = Math.max(0, parseInt(req.query.offset) || 0);
  const limit = Math.min(50, Math.max(1, parseInt(req.query.limit) || 10));
  try {
    const dir = safeDirPath(req.query.path || '');
    fs.mkdirSync(dir, { recursive: true });
    const all = fs.readdirSync(dir)
      .filter(f => IMAGE_RE.test(f))
      .map(f => {
        const stat = fs.statSync(path.join(dir, f));
        return { name: f, size: stat.size, modified: stat.mtime };
      })
      .sort((a, b) => new Date(b.modified) - new Date(a.modified));
    res.json({ photos: all.slice(offset, offset + limit), total: all.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.get('/file', (req, res) => {
  const dir = safeDirPath(req.query.path || '');
  const fp = path.join(dir, path.basename(String(req.query.name || '')));
  if (!fp.startsWith(config.photoDir) || !fs.existsSync(fp)) {
    return res.status(404).json({ error: 'Not found' });
  }
  res.sendFile(fp);
});

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const dir = safeDirPath(req.query.path || '');
      fs.mkdirSync(dir, { recursive: true });
      cb(null, dir);
    },
    filename: (_req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
  }),
});

router.post('/upload', upload.array('photos', 50), (req, res) => {
  if (!req.files?.length) return res.status(400).json({ error: 'No files uploaded' });
  res.json({ uploaded: req.files.map(f => ({ name: f.filename, size: f.size })) });
});

router.post('/copy', (req, res) => {
  const { from_path, from_name, to_path } = req.body;
  const name = path.basename(String(from_name || ''));
  const fromFile = path.join(safeDirPath(from_path || ''), name);
  const toDir = safeDirPath(to_path || '');
  const toFile = path.join(toDir, `${Date.now()}-${name}`);
  if (!fromFile.startsWith(config.photoDir) || !fs.existsSync(fromFile))
    return res.status(404).json({ error: 'Source not found' });
  if (!toFile.startsWith(config.photoDir))
    return res.status(400).json({ error: 'Invalid destination' });
  fs.mkdirSync(toDir, { recursive: true });
  fs.copyFileSync(fromFile, toFile);
  res.json({ ok: true });
});

router.delete('/files', (req, res) => {
  const { names } = req.body;
  const dir = safeDirPath(req.body.path || '');
  if (!Array.isArray(names)) return res.status(400).json({ error: 'names must be an array' });
  const deleted = names.filter(n => {
    const fp = path.join(dir, path.basename(String(n)));
    if (!fp.startsWith(config.photoDir) || !fs.existsSync(fp)) return false;
    fs.unlinkSync(fp);
    return true;
  });
  res.json({ deleted });
});

module.exports = router;
