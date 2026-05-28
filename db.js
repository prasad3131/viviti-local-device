const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');
const config = require('./config');

fs.mkdirSync(config.dataDir, { recursive: true });

const db = new Database(path.join(config.dataDir, 'viviti.db'));

db.exec(`
  CREATE TABLE IF NOT EXISTS photo_ai (
    photo_path   TEXT PRIMARY KEY,
    processed_at TEXT DEFAULT (datetime('now')),
    blur_score   REAL,
    is_blurry    INTEGER DEFAULT 0,
    phash        TEXT,
    is_duplicate INTEGER DEFAULT 0,
    duplicate_of TEXT,
    scene_tags   TEXT,
    objects      TEXT,
    ai_score     REAL
  )
`);

// Add columns when upgrading from older schema
for (const col of ['scene_tags TEXT', 'objects TEXT']) {
  try { db.exec(`ALTER TABLE photo_ai ADD COLUMN ${col}`); } catch {}
}

module.exports = db;
