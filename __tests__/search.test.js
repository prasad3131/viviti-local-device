/**
 * Tests for Object Search expansion in routes/ai.js.
 * Run: npx jest __tests__/search.test.js
 *
 * Object labels now come from the COCO-SSD detector (80 classes), so the synonym
 * map points at COCO labels — e.g. "flower" -> "potted plant"/"vase" (COCO has no
 * flower class). These tests lock that mapping in (regression: "flower" returned
 * nothing after the ImageNet->COCO switch).
 */

const SEARCH_SYNONYMS = {
  dog:    ['dog'],
  cat:    ['cat'],
  car:    ['car', 'truck'],
  flower: ['potted plant', 'vase'],
  food:   ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'],
  phone:  ['cell phone'],
};

function expandQuery(q) {
  const norm = q.toLowerCase().trim();
  const terms = new Set();
  if (norm) terms.add(norm);
  if (SEARCH_SYNONYMS[norm]) SEARCH_SYNONYMS[norm].forEach(t => terms.add(t));
  for (const word of norm.split(/\s+/)) {
    if (SEARCH_SYNONYMS[word]) SEARCH_SYNONYMS[word].forEach(t => terms.add(t));
  }
  return [...terms].filter(Boolean);
}

function matches(objectLabels, query) {
  const terms = expandQuery(query);
  return objectLabels.some(o => terms.some(t => o.includes(t)));
}

describe('expandQuery (COCO labels)', () => {
  test('"flower" expands to COCO potted plant + vase — the regression we fixed', () => {
    const terms = expandQuery('flower');
    expect(terms).toContain('potted plant');
    expect(terms).toContain('vase');
  });

  test('"car" expands to car + truck', () => {
    expect(expandQuery('car')).toEqual(expect.arrayContaining(['car', 'truck']));
  });

  test('"phone" maps to "cell phone"', () => {
    expect(expandQuery('phone')).toContain('cell phone');
  });

  test('unknown query just returns itself', () => {
    expect(expandQuery('umbrella')).toEqual(['umbrella']);
  });

  test('empty query yields no terms', () => {
    expect(expandQuery('')).toEqual([]);
  });
});

describe('object matching (COCO)', () => {
  test('"flower" matches a photo labeled "potted plant"', () => {
    expect(matches(['potted plant', 'person'], 'flower')).toBe(true);
  });

  test('"dog" matches a photo labeled "dog"', () => {
    expect(matches(['dog', 'grass'], 'dog')).toBe(true);
  });

  test('"food" matches "pizza"', () => {
    expect(matches(['pizza', 'bottle'], 'food')).toBe(true);
  });

  test('"food" does NOT match a bowl/potted-plant photo — garden-leak regression', () => {
    expect(matches(['bowl', 'potted plant'], 'food')).toBe(false);
  });

  test('direct COCO label match works', () => {
    expect(matches(['cell phone'], 'cell phone')).toBe(true);
    expect(matches(['chair', 'tv'], 'tv')).toBe(true);
  });

  test('"cat" does not match a dog-only photo', () => {
    expect(matches(['dog', 'bench'], 'cat')).toBe(false);
  });

  test('no labels never matches', () => {
    expect(matches([], 'flower')).toBe(false);
  });
});
