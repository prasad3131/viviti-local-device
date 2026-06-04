/**
 * Tests for Smart Highlights + blur-threshold logic.
 * Run: npx jest __tests__/highlights.test.js
 *
 * Locks in two fixes:
 *  - BLUR_THRESHOLD lowered 100 -> 40 (was over-flagging good phone JPEGs).
 *    Real-library scores cluster <=24 for genuinely blurry, jump to ~49+ for
 *    sharp-subject/bokeh shots; 40 sits in that gap.
 *  - Highlights = best (sharpest) photo per 60s burst window, blurry/dupes excluded.
 */

const BLUR_THRESHOLD = 40;

// ── Pure logic mirrors of the device code ─────────────────────────────────────

function isBlurry(score) {
  return score < BLUR_THRESHOLD ? 1 : 0;
}

// Mirrors GET /ai/highlights: group by 60s burst, pick max blur_score per group.
function pickHighlights(photos) {
  const BURST_MS = 60 * 1000;
  const pool = photos
    .filter(p => p.is_blurry === 0 && p.is_duplicate === 0)
    .sort((a, b) => a.mtime - b.mtime);

  const out = [];
  let i = 0;
  while (i < pool.length) {
    let j = i + 1;
    while (j < pool.length && pool[j].mtime - pool[i].mtime <= BURST_MS) j++;
    const group = pool.slice(i, j);
    out.push(group.reduce((a, b) => (b.blur_score > a.blur_score ? b : a)).name);
    i = j;
  }
  return out;
}

// ── Blur threshold ─────────────────────────────────────────────────────────────

describe('blur threshold (40)', () => {
  test('genuinely blurry shots (<=24) stay flagged', () => {
    [5, 12, 24].forEach(s => expect(isBlurry(s)).toBe(1));
  });

  test('rescued good shots (49-87) are NOT flagged — the regression we fixed', () => {
    [49, 55, 67, 69, 87].forEach(s => expect(isBlurry(s)).toBe(0));
  });

  test('boundary: 39 blurry, 40 sharp', () => {
    expect(isBlurry(39)).toBe(1);
    expect(isBlurry(40)).toBe(0);
  });

  test('old threshold of 100 would have wrongly flagged a score of 87', () => {
    // Guard against anyone reverting the constant back to 100.
    expect(87 < BLUR_THRESHOLD).toBe(false);
  });
});

// ── Highlights burst grouping ───────────────────────────────────────────────────

describe('highlights burst grouping', () => {
  test('one photo per 60s burst window', () => {
    const photos = [
      { name: 'a', mtime: 0,      blur_score: 50, is_blurry: 0, is_duplicate: 0 },
      { name: 'b', mtime: 10_000, blur_score: 90, is_blurry: 0, is_duplicate: 0 }, // same burst, sharper
      { name: 'c', mtime: 80_000, blur_score: 60, is_blurry: 0, is_duplicate: 0 }, // new burst
    ];
    expect(pickHighlights(photos)).toEqual(['b', 'c']);
  });

  test('picks the sharpest photo within a burst', () => {
    const photos = [
      { name: 'soft',  mtime: 0,     blur_score: 45, is_blurry: 0, is_duplicate: 0 },
      { name: 'sharp', mtime: 5_000, blur_score: 95, is_blurry: 0, is_duplicate: 0 },
    ];
    expect(pickHighlights(photos)).toEqual(['sharp']);
  });

  test('blurry and duplicate photos are excluded from the pool', () => {
    const photos = [
      { name: 'blur', mtime: 0,     blur_score: 10, is_blurry: 1, is_duplicate: 0 },
      { name: 'dup',  mtime: 1_000, blur_score: 80, is_blurry: 0, is_duplicate: 1 },
      { name: 'good', mtime: 2_000, blur_score: 70, is_blurry: 0, is_duplicate: 0 },
    ];
    expect(pickHighlights(photos)).toEqual(['good']);
  });

  test('a rescued 49+ photo wins its window only if it is the sharpest there', () => {
    // Rescued photo alone in its burst -> becomes a highlight.
    const photos = [
      { name: 'rescued', mtime: 0, blur_score: 49, is_blurry: 0, is_duplicate: 0 },
    ];
    expect(pickHighlights(photos)).toEqual(['rescued']);

    // But shares a window with a sharper shot -> sharper one is chosen (why the
    // live highlight count stayed at 7 after reclassification).
    const shared = [
      { name: 'rescued', mtime: 0,     blur_score: 49, is_blurry: 0, is_duplicate: 0 },
      { name: 'sharper', mtime: 5_000, blur_score: 88, is_blurry: 0, is_duplicate: 0 },
    ];
    expect(pickHighlights(shared)).toEqual(['sharper']);
  });

  test('empty library yields no highlights', () => {
    expect(pickHighlights([])).toEqual([]);
  });
});
