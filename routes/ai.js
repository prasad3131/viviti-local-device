const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const config = require('../config');

const CRITIQUE_SCRIPT = path.join(__dirname, '..', 'ai', 'critique.py');
const BATCH_SCRIPT    = path.join(__dirname, '..', 'ai', 'batch.py');
const DB_PATH         = path.join(config.dataDir, 'viviti.db');

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

module.exports = { router, triggerBatch };
