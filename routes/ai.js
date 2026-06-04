const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const config = require('../config');

const CRITIQUE_SCRIPT      = path.join(__dirname, '..', 'ai', 'critique.py');
const BATCH_SCRIPT         = path.join(__dirname, '..', 'ai', 'batch.py');
const FACES_SCRIPT         = path.join(__dirname, '..', 'ai', 'faces.py');
const DETECT_PHOTO_SCRIPT  = path.join(__dirname, '..', 'ai', 'detect_photo.py');
const OBJDETECT_SCRIPT     = path.join(__dirname, '..', 'ai', 'objdetect.py');
const DB_PATH         = path.join(config.dataDir, 'viviti.db');
const FACE_THUMB_DIR  = path.join(config.dataDir, 'face_thumbs');

// Use venv Python if available (needed for mediapipe on Python 3.13 systems)
const PYTHON = fs.existsSync('/opt/ai-env/bin/python3')
  ? '/opt/ai-env/bin/python3'
  : 'python3';

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
    const py = spawn(PYTHON, [scriptPath, ...args]);
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

// POST /ai/faces/detect-photo { path, name } — detect faces in one photo on-demand
router.post('/faces/detect-photo', async (req, res) => {
  const fp = safePath(req.body.path, req.body.name);
  if (!fp) return res.status(400).json({ error: 'Invalid path' });
  if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Photo not found' });

  const db = require('../db');
  const relPath = path.relative(config.photoDir, fp).replace(/\\/g, '/');

  // Use batch-scan results if already in the DB — those ran through the full quality filters
  let existing;
  try {
    existing = db.prepare(`
      SELECT pf.x, pf.y, pf.w, pf.h, pf.thumb_path, pf.cluster_id,
             fc.name AS cluster_name
      FROM photo_faces pf
      LEFT JOIN face_clusters fc ON fc.id = pf.cluster_id
      WHERE pf.photo_path = ?
    `).all(relPath);
  } catch { existing = []; }

  if (existing.length > 0) {
    // Deduplicate by cluster_id — consolidate_clusters can merge two clusters
    // that both appeared in this photo, producing duplicate cluster_id rows.
    const seen = new Set();
    const deduped = existing.filter(r => {
      if (seen.has(r.cluster_id)) return false;
      seen.add(r.cluster_id);
      return true;
    });
    return res.json({
      faces: deduped.map(r => ({
        x: r.x, y: r.y, w: r.w, h: r.h,
        cluster_id: r.cluster_id,
        cluster_name: r.cluster_name,
        thumb_filename: r.thumb_path ? path.basename(r.thumb_path) : null,
      })),
    });
  }

  // Photo not yet scanned — run fresh detection and store results
  try {
    const result = await runPython(DETECT_PHOTO_SCRIPT, [fp, DB_PATH, config.photoDir], 30_000);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

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

// PUT /ai/faces/:id — set name (and optionally anchor the thumbnail) for a cluster
router.put('/faces/:id', (req, res) => {
  const db = require('../db');
  const name = String(req.body.name || '').trim();
  if (!name) return res.status(400).json({ error: 'name required' });
  // When the user names a face from the photo viewer, pin sample_thumb to the
  // thumbnail they were looking at so People screen matches their expectation.
  const rawThumb = req.body.thumb_filename ? String(req.body.thumb_filename) : null;
  const thumbPath = rawThumb ? path.join(FACE_THUMB_DIR, path.basename(rawThumb)) : null;
  if (thumbPath) {
    db.prepare('UPDATE face_clusters SET name = ?, sample_thumb = ? WHERE id = ?')
      .run(name, thumbPath, req.params.id);
  } else {
    db.prepare('UPDATE face_clusters SET name = ? WHERE id = ?').run(name, req.params.id);
  }
  res.json({ ok: true });
});

// DELETE /ai/faces/:id — delete a face cluster and all its face records
router.delete('/faces/:id', (req, res) => {
  const db = require('../db');
  db.prepare('DELETE FROM photo_faces WHERE cluster_id = ?').run(req.params.id);
  db.prepare('DELETE FROM face_clusters WHERE id = ?').run(req.params.id);
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

// ── Auto-Albums ───────────────────────────────────────────────────────────────

const IMAGE_RE_ALBUMS = /\.(jpg|jpeg|png|gif|heic|raw|cr2|arw|nef|dng)$/i;

function walkDir(dir, cb) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walkDir(full, cb);
    else if (IMAGE_RE_ALBUMS.test(e.name)) cb(full, e.name);
  }
}

// GET /ai/albums — list all months that have photos
router.get('/albums', (req, res) => {
  const albums = {};
  try {
    walkDir(config.photoDir, (full, name) => {
      const stat = fs.statSync(full);
      const d = new Date(stat.mtime);
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
      const rel = path.relative(config.photoDir, full).replace(/\\/g, '/');
      if (!albums[key]) albums[key] = { key, cover: rel, count: 0 };
      albums[key].count++;
    });
  } catch {}
  const list = Object.values(albums).sort((a, b) => b.key.localeCompare(a.key));
  res.json({ albums: list });
});

// GET /ai/albums/:key/photos — photos in a specific month (key = "YYYY-MM")
router.get('/albums/:key/photos', (req, res) => {
  const key = String(req.params.key);
  if (!/^\d{4}-\d{2}$/.test(key)) return res.status(400).json({ error: 'invalid key' });
  const [year, month] = key.split('-').map(Number);
  const photos = [];
  try {
    walkDir(config.photoDir, (full) => {
      const stat = fs.statSync(full);
      const d = new Date(stat.mtime);
      if (d.getFullYear() !== year || d.getMonth() + 1 !== month) return;
      const rel = path.relative(config.photoDir, full).replace(/\\/g, '/');
      const i = rel.lastIndexOf('/');
      photos.push({
        photo_path: rel,
        folder: i >= 0 ? rel.slice(0, i) : '',
        name: i >= 0 ? rel.slice(i + 1) : rel,
        mtime: stat.mtime,
      });
    });
  } catch {}
  photos.sort((a, b) => new Date(b.mtime) - new Date(a.mtime));
  res.json({ photos });
});

// ── Smart Highlights ──────────────────────────────────────────────────────────

// GET /ai/highlights — best photo from each burst window (60s)
router.get('/highlights', (req, res) => {
  const db = require('../db');
  let rows;
  try {
    rows = db.prepare(
      'SELECT photo_path, blur_score FROM photo_ai WHERE is_blurry = 0 AND is_duplicate = 0 ORDER BY photo_path'
    ).all();
  } catch {
    return res.json({ highlights: [] });
  }

  const BURST_MS = 60 * 1000;
  const withTimes = rows.map(r => {
    try {
      const abs = path.join(config.photoDir, r.photo_path.replace(/\//g, path.sep));
      const stat = fs.statSync(abs);
      return { photo_path: r.photo_path, blur_score: r.blur_score, mtime: stat.mtimeMs };
    } catch { return null; }
  }).filter(Boolean).sort((a, b) => a.mtime - b.mtime);

  const highlights = [];
  let i = 0;
  while (i < withTimes.length) {
    let j = i + 1;
    while (j < withTimes.length && withTimes[j].mtime - withTimes[i].mtime <= BURST_MS) j++;
    const group = withTimes.slice(i, j);
    const best = group.reduce((a, b) => b.blur_score > a.blur_score ? b : a);
    highlights.push(best.photo_path);
    i = j;
  }

  const result = highlights.map(p => {
    const idx = p.lastIndexOf('/');
    return { photo_path: p, folder: idx >= 0 ? p.slice(0, idx) : '', name: idx >= 0 ? p.slice(idx + 1) : p };
  });
  res.json({ highlights: result });
});

// ── Object Search ───────────────────────────────────────────────────────────
// Searches the `objects` column (top-5 ImageNet labels per photo, written by the
// scene pass) + `scene_tags`. No new model, no network — pure on-device lookup.

// User-friendly query -> COCO-SSD object labels (+ scene tags) that should match.
// Objects come from the COCO detector (80 classes), so e.g. "flower" must map to
// "potted plant"/"vase" (COCO has no flower class). Scene words (beach, nature,
// outdoor, ...) also match the scene_tags column via the raw query.
const SEARCH_SYNONYMS = {
  // animals
  dog:       ['dog'],
  puppy:     ['dog'],
  cat:       ['cat'],
  kitten:    ['cat'],
  bird:      ['bird'],
  horse:     ['horse'],
  cow:       ['cow'],
  sheep:     ['sheep'],
  animal:    ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe'],
  pet:       ['dog', 'cat', 'bird'],
  // vehicles
  car:       ['car', 'truck'],
  truck:     ['truck'],
  bus:       ['bus'],
  bike:      ['bicycle', 'motorcycle'],
  bicycle:   ['bicycle'],
  motorcycle:['motorcycle'],
  train:     ['train'],
  boat:      ['boat'],
  plane:     ['airplane'],
  airplane:  ['airplane'],
  vehicle:   ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane', 'boat'],
  // plants / nature
  flower:    ['potted plant', 'vase'],
  plant:     ['potted plant'],
  // food & drink — only actual edible items (no 'bowl': it's tableware and
  // matches plant pots, leaking garden photos into "food" results).
  food:      ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
  fruit:     ['banana', 'apple', 'orange'],
  pizza:     ['pizza'],
  cake:      ['cake'],
  drink:     ['bottle', 'wine glass', 'cup'],
  bottle:    ['bottle'],
  // people / things
  baby:      ['person'],
  people:    ['person'],
  phone:     ['cell phone'],
  laptop:    ['laptop'],
  computer:  ['laptop', 'keyboard', 'mouse', 'tv'],
  tv:        ['tv'],
  television:['tv'],
  furniture: ['chair', 'couch', 'bed', 'dining table', 'bench'],
  couch:     ['couch'],
  sofa:      ['couch'],
  bag:       ['backpack', 'handbag', 'suitcase'],
  toy:       ['teddy bear'],
  teddy:     ['teddy bear'],
  // sports
  sport:     ['sports ball', 'tennis racket', 'baseball bat', 'skateboard', 'surfboard', 'frisbee', 'skis', 'snowboard', 'baseball glove', 'kite'],
  ball:      ['sports ball'],
};

function expandQuery(q) {
  const norm  = q.toLowerCase().trim();
  const terms = new Set();
  if (norm) terms.add(norm);
  if (SEARCH_SYNONYMS[norm]) SEARCH_SYNONYMS[norm].forEach(t => terms.add(t));
  for (const word of norm.split(/\s+/)) {
    if (SEARCH_SYNONYMS[word]) SEARCH_SYNONYMS[word].forEach(t => terms.add(t));
  }
  return [...terms].filter(Boolean);
}

// GET /ai/search?q=dog&path=folder
// Matches objects + scene tags AND people's names (the names assigned to faces).
router.get('/search', (req, res) => {
  const db = require('../db');
  const q  = String(req.query.q || '').trim();
  if (!q) return res.json({ photos: [], terms: [] });

  const terms  = expandQuery(q);
  const prefix = req.query.path ? String(req.query.path) + '/' : '';

  // photo_path -> { matched: string[], person: string|null }
  const hits = new Map();
  const add = (photo_path, label, isPerson) => {
    let e = hits.get(photo_path);
    if (!e) { e = { matched: [], person: null }; hits.set(photo_path, e); }
    if (isPerson) e.person = label;
    if (label && !e.matched.includes(label)) e.matched.push(label);
  };

  // 1. Object / scene matches
  try {
    const likes = [];
    const params = [];
    for (const t of terms) {
      likes.push('objects LIKE ?');    params.push(`%${t}%`);
      likes.push('scene_tags LIKE ?'); params.push(`%${t}%`);
    }
    params.push(prefix + '%');
    const rows = db.prepare(
      `SELECT photo_path, objects FROM photo_ai
       WHERE (${likes.join(' OR ')}) AND photo_path LIKE ?
       ORDER BY photo_path DESC LIMIT 500`
    ).all(...params);
    for (const { photo_path, objects } of rows) {
      let matched = [];
      try { matched = JSON.parse(objects || '[]').filter(o => terms.some(t => o.includes(t))); } catch {}
      matched.slice(0, 3).forEach(m => add(photo_path, m, false));
      if (matched.length === 0) add(photo_path, null, false);
    }
  } catch {}

  // 2. People-name matches — names the user assigned to face clusters
  try {
    const people = db.prepare(
      `SELECT DISTINCT pf.photo_path AS photo_path, fc.name AS name
       FROM photo_faces pf JOIN face_clusters fc ON fc.id = pf.cluster_id
       WHERE fc.name IS NOT NULL AND fc.name != '' AND fc.name LIKE ? AND pf.photo_path LIKE ?
       LIMIT 500`
    ).all(`%${q}%`, prefix + '%');
    for (const { photo_path, name } of people) add(photo_path, name, true);
  } catch {}

  const photos = [...hits.entries()].map(([photo_path, info]) => {
    const i = photo_path.lastIndexOf('/');
    return {
      photo_path,
      folder: i >= 0 ? photo_path.slice(0, i) : '',
      name:   i >= 0 ? photo_path.slice(i + 1) : photo_path,
      matched_objects: info.matched.slice(0, 3),
      person: info.person || undefined,
    };
  });
  res.json({ photos, terms });
});

// GET /ai/objects — distinct object labels with counts (browse + suggestions)
router.get('/objects', (_req, res) => {
  const db = require('../db');
  let rows;
  try {
    rows = db.prepare(`SELECT objects FROM photo_ai WHERE objects IS NOT NULL AND objects != '[]'`).all();
  } catch {
    return res.json({ objects: [] });
  }
  const counts = {};
  for (const { objects } of rows) {
    try { for (const o of JSON.parse(objects)) counts[o] = (counts[o] || 0) + 1; } catch {}
  }
  const list = Object.entries(counts)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
  res.json({ objects: list });
});

// POST /ai/objects/rescan — re-run object detection across the whole library and
// update the `objects` column (leaves blur/dup/scene data untouched). Background.
let objRescanRunning = false;
router.post('/objects/rescan', (_req, res) => {
  res.json({ ok: true, already_running: objRescanRunning });
  if (objRescanRunning) return;
  objRescanRunning = true;
  console.log('[AI] Object re-scan started');
  runPython(OBJDETECT_SCRIPT, ['--batch', config.photoDir, DB_PATH], 30 * 60_000)
    .then(r  => console.log('[AI] Object re-scan done:', r))
    .catch(e => console.error('[AI] Object re-scan error:', e.message))
    .finally(() => { objRescanRunning = false; });
});

module.exports = { router, triggerBatch };
