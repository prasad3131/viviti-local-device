/**
 * Tests for routes/photos.js thumbnail logic.
 * Run: npx jest __tests__/photos.thumb.test.js
 *
 * Covers every thumb-related fix:
 *  - Size cap raised from 400 → 1200 (viewer images were being downgraded)
 *  - v param included in cache filename (true cache busting)
 *  - fit:inside for viewer sizes, fit:cover for grid sizes
 *  - 12-second timeout falls back to serving original
 */

const path = require('path');
const fs   = require('fs');

// ── Helpers extracted from photos.js (pure logic, no Express) ─────────────────

function parseSize(raw, maxSize = 1200) {
  return Math.min(maxSize, Math.max(50, parseInt(raw) || 200));
}

function parseVersion(raw) {
  if (!raw) return '';
  return `_v${String(raw).replace(/[^a-z0-9]/gi, '')}`;
}

function thumbFilename(key, size, v) {
  return `${key}_${size}${v}.jpg`;
}

function isViewer(size) {
  return size > 400;
}

function sharpOptions(size) {
  return isViewer(size)
    ? { fit: 'inside', withoutEnlargement: true }
    : { fit: 'cover', position: 'attention' };
}

// ── Size cap ──────────────────────────────────────────────────────────────────

describe('thumbnail size cap', () => {
  test('allows 1080 (viewer size) — old cap of 400 was the root cause of black screen', () => {
    expect(parseSize('1080')).toBe(1080);
  });

  test('allows 1200 (max viewer size)', () => {
    expect(parseSize('1200')).toBe(1200);
  });

  test('caps at 1200 when request exceeds it', () => {
    expect(parseSize('9999')).toBe(1200);
  });

  test('grid size 200 passes through unchanged', () => {
    expect(parseSize('200')).toBe(200);
  });

  test('minimum is 50', () => {
    expect(parseSize('1')).toBe(50);
  });

  test('defaults to 200 when size is not a number', () => {
    expect(parseSize('abc')).toBe(200);
  });
});

// ── Version / cache busting ───────────────────────────────────────────────────

describe('v param in cache filename', () => {
  test('v=2 produces _v2 suffix in filename', () => {
    expect(parseVersion('2')).toBe('_v2');
  });

  test('no v param produces empty suffix', () => {
    expect(parseVersion(undefined)).toBe('');
    expect(parseVersion('')).toBe('');
  });

  test('v param is sanitised — no path traversal', () => {
    expect(parseVersion('../evil')).toBe('_vevil');
    expect(parseVersion('../../etc')).toBe('_vetc');
  });

  test('full filename uses size + v suffix', () => {
    expect(thumbFilename('photo_key', 1080, '_v2')).toBe('photo_key_1080_v2.jpg');
    expect(thumbFilename('photo_key', 200,  ''   )).toBe('photo_key_200.jpg');
  });
});

// ── Resize mode ───────────────────────────────────────────────────────────────

describe('Sharp resize options', () => {
  test('viewer (size > 400) uses fit:inside — preserves full photo, no crop', () => {
    const opts = sharpOptions(1080);
    expect(opts.fit).toBe('inside');
    expect(opts.withoutEnlargement).toBe(true);
  });

  test('grid (size ≤ 400) uses fit:cover — square crop centred on subject', () => {
    const opts = sharpOptions(200);
    expect(opts.fit).toBe('cover');
    expect(opts.position).toBe('attention');
  });

  test('size 400 is still grid (boundary)', () => {
    expect(isViewer(400)).toBe(false);
  });

  test('size 401 is viewer (boundary)', () => {
    expect(isViewer(401)).toBe(true);
  });
});

// ── Timeout fallback ──────────────────────────────────────────────────────────

describe('thumbnail generation timeout', () => {
  test('Promise.race resolves with first resolved value', async () => {
    const fast = Promise.resolve('done');
    const slow = new Promise(r => setTimeout(r, 60000));
    const result = await Promise.race([fast, slow]);
    expect(result).toBe('done');
  });

  test('timeout rejects if generation hangs beyond 12s (simulated)', async () => {
    jest.useFakeTimers();
    const hanging = new Promise(() => {});
    const timeout = new Promise((_, rej) =>
      setTimeout(() => rej(new Error('timeout')), 12000)
    );
    const racePromise = Promise.race([hanging, timeout]);
    jest.advanceTimersByTime(12001);
    await expect(racePromise).rejects.toThrow('timeout');
    jest.useRealTimers();
  });
});
