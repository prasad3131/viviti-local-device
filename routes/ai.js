const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const config = require('../config');

const CRITIQUE_SCRIPT = path.join(__dirname, '..', 'ai', 'critique.py');
const BATCH_SCRIPT    = path.join(__dirname, '..', 'ai', 'batch.py');
const FACES_SCRIPT    = path.join(__dirname, '..', 'ai', 'faces.py');
const DB_PATH         = path.join(config.dataDir, 'viviti.db');
const FACE_THUMB_DIR  = path.join(config.dataDir, 'face_thumbs');

function safePath(userPath, name) {
  const parts = String(userPath || '').split('/').map(p => path.basename(p)).filter(Boolean);
  const dir = path.join(config.photoDir, ...parts);
  if (!dir.startsWith(config.photoDir)) return null;
  if (!name) return dir;
  const fp = path.join(dir, path.basename(String(name)));
  return fp.startsWith(config.photoDir) ? fp : null;
}

function runPython(scriptPath, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', [scriptPath, ...args]);
    let out = '', err = '';
    const timer = setTimeout(() => { py.kill(); reject(new Error('Python timeout')); }, timeoutMs);
    py.stdout.on('data', d => { out += d; });
    py.stderr.on('data', d => { err += d; });
    py.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(err.trim() || `exited ${code}`));
      try { resolve(JSON.parse(out)); } catch { reject(new Error('Invalid output from Python')); }
    });
  });
}

// ── Photo critique (on-demand) ───────────────────────────────────────────────
// POST /ai/critique  { path, name }
router.post('/critique', async (req, res) => {
  const fp = safePath(req.body.path, req.body.name);
  if (!fp) return res.status(400).json({ error: 'Invalid path' });
  if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Photo not found' });
  try {
    const result = await runPython(CRITIQUE_SCRIPT, [fp], 15_000);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Batch processing ─────────────────────────────────────────────────────────
let batchRunning = false;

async function triggerBatch() {
  if (batchRunning) return;
  batchRunning = true;
  console.log('[AI] Batch started');
  try {
    const result = await runPython(BATCH_SCRIPT, [config.photoDir, DB_PATH], 30 * 60_000);
    console.log('[AI] Batch done:', result);
  } catch (e) {
    console.error('[AI] Batch error:', e.message);
  } finally {
    batchRunning = false;
  }
}

// POST /ai/batch/run  — manual trigger (returns immediately, runs in background)
router.post('/batch/run', (req, res) => {
  res.json({ ok: true, already_running: batchRunning });
  triggerBatch();
});

// GET /ai/status
router.get('/status', (_req, res) => {
  const db = require('../db');
  const stats = db.prepare(`
    SELECT
      COUNT(*)                                            AS total,
      COALESCE(SUM(is_blurry),    0)                    AS blurry,
      COALESCE(SUM(is_duplicate), 0)                    AS duplicates
    FROM photo_ai
  `).get();
  res.json({ ...stats, batch_running: batchRunning });
});

// GET /ai/blurry?path=folder
router.get('/blurry', (req, res) => {
  const db = require('../db');
  const prefix = req.query.path ? String(req.query.path).replace(/\//g, path.sep) + path.sep : '';
  const rows = db.prepare(
    `SELECT photo_path, blur_score FROM photo_ai WHERE is_blurry = 1 AND photo_path LIKE ? ORDER BY blur_score ASC`
  ).all(prefix.replace(/\\/g, '/') + '%');
  res.json({ photos: rows });
});

// GET /ai/duplicates?path=folder
router.get('/duplicates', (req, res) => {
  const db = require('../db');
  const prefix = req.query.path ? String(req.query.path) + '/' : '';
  const rows = db.prepare(
    `SELECT photo_path, duplicate_of FROM photo_ai WHERE is_duplicate = 1 AND photo_path LIKE ? ORDER BY duplicate_of`
  ).all(prefix + '%');
  res.json({ photos: rows });
});

// GET /ai/tags?path=folder  — tags + counts for photos in folder
router.get('/tags', (req, res) => {
  const db = require('../db');
  const prefix = req.query.path ? String(req.query.path) + '/' : '';
  const rows = db.prepare(
    `SELECT scene_tags FROM photo_ai WHERE scene_tags IS NOT NULL AND scene_tags != '[]' AND photo_path LIKE ?`
  ).all(prefix + '%');

  const counts = {};
  for (const { scene_tags } of rows) {
    try {
      for (const tag of JSON.parse(scene_tags)) {
        counts[tag] = (counts[tag] || 0) + 1;
      }
    } catch {}
  }
  const tags = Object.entries(counts)
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count);
  res.json({ tags });
});

// GET /ai/tagged?tag=beach&path=folder  — photos with a specific tag
router.get('/tagged', (req, res) => {
  const db = require('../db');
  const tag    = String(req.query.tag || '');
  const prefix = req.query.path ? String(req.query.path) + '/' : '';
  if (!tag) return res.status(400).json({ error: 'tag required' });

  const rows = db.prepare(
    `SELECT photo_path FROM photo_ai WHERE scene_tags LIKE ? AND photo_path LIKE ?`
  ).all(`%"${tag}"%`, prefix + '%');

  const photos = rows.map(({ photo_path }) => {
    const lastSlash = photo_path.lastIndexOf('/');
    return {
      photo_path,
      folder: lastSlash >= 0 ? photo_path.slice(0, lastSlash) : '',
      name:   lastSlash >= 0 ? photo_path.slice(lastSlash + 1) : photo_path,
    };
  });
  res.json({ photos });
});

// ── Face recognition ─────────────────────────────────────────────────────────

// POST /ai/faces/run — trigger face batch
let facesRunning = false;
router.post('/faces/run', (_req, res) => {
  res.json({ ok: true, already_running: facesRunning });
  if (facesRunning) return;
  facesRunning = true;
  console.log('[AI] Face batch started');
  runPython(FACES_SCRIPT, [config.photoDir, DB_PATH], 30 * 60_000)
    .then(r  => console.log('[AI] Face batch done:', r))
    .catch(e => console.error('[AI] Face batch error:', e.message))
    .finally(() => { facesRunning = false; });
});

// GET /ai/faces — list all face clusters
router.get('/faces', (_req, res) => {
  const db = require('../db');
  const rows = db.prepare(
    'SELECT id, name, sample_thumb, photo_count FROM face_clusters ORDER BY photo_count DESC'
  ).all();
  res.json({ faces: rows });
});

// PUT /ai/faces/:id — set name for a cluster
router.put('/faces/:id', (req, res) => {
  const db = require('../db');
  const name = String(req.body.name || '').trim();
  if (!name) return res.status(400).json({ error: 'name required' });
  db.prepare('UPDATE face_clusters SET name = ? WHERE id = ?').run(name, req.params.id);
  res.json({ ok: true });
});

// GET /ai/faces/:id/photos — photos containing this person
router.get('/faces/:id/photos', (req, res) => {
  const db = require('../db');
  const rows = db.prepare(
    'SELECT DISTINCT photo_path FROM photo_faces WHERE cluster_id = ? ORDER BY photo_path'
  ).all(req.params.id);
  const photos = rows.map(({ photo_path }) => {
    const i = photo_path.lastIndexOf('/');
    return { photo_path, folder: i >= 0 ? photo_path.slice(0, i) : '', name: i >= 0 ? photo_path.slice(i + 1) : photo_path };
  });
  res.json({ photos });
});

// GET /ai/faces/thumb/:filename — serve face thumbnail
router.get('/faces/thumb/:filename', (req, res) => {
  const fp = path.join(FACE_THUMB_DIR, path.basename(req.params.filename));
  if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Not found' });
  res.setHeader('Cache-Control', 'public, max-age=604800');
  res.sendFile(fp);
});

module.exports = { router, triggerBatch };
