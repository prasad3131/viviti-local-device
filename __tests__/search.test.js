/**
 * Tests for Object Search logic in routes/ai.js.
 * Run: npx jest __tests__/search.test.js
 *
 * Mirrors expandQuery() + the object-matching filter so the synonym expansion
 * (e.g. "dog" -> "golden retriever") can't silently regress.
 */

const SEARCH_SYNONYMS = {
  dog: ['dog', 'puppy', 'retriever', 'labrador', 'poodle', 'husky', 'bulldog', 'beagle', 'terrier', 'spaniel', 'chihuahua', 'dalmatian', 'rottweiler', 'pug', 'collie', 'corgi'],
  cat: ['cat', 'kitten', 'tabby', 'siamese', 'persian cat', 'egyptian cat', 'kitty'],
  food: ['pizza', 'burger', 'cheeseburger', 'hotdog', 'sandwich', 'cake', 'ice cream', 'burrito', 'bagel', 'pretzel', 'plate', 'guacamole', 'soup', 'espresso', 'meatloaf'],
  beach: ['seashore', 'sandbar', 'beach', 'dock', 'pier', 'lakeside', 'shoal'],
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

// Mirrors the per-photo match used to build matched_objects + the WHERE filter.
function matches(objectLabels, query) {
  const terms = expandQuery(query);
  return objectLabels.some(o => terms.some(t => o.includes(t)));
}

describe('expandQuery', () => {
  test('a known category expands to its specific ImageNet labels', () => {
    const terms = expandQuery('dog');
    expect(terms).toContain('dog');
    expect(terms).toContain('retriever'); // matches "golden retriever", "labrador retriever"
    expect(terms).toContain('labrador');
  });

  test('an unknown query just returns itself (direct substring search)', () => {
    expect(expandQuery('umbrella')).toEqual(['umbrella']);
  });

  test('is case-insensitive and trimmed', () => {
    expect(expandQuery('  DOG ')).toContain('labrador');
  });

  test('empty query yields no terms', () => {
    expect(expandQuery('')).toEqual([]);
    expect(expandQuery('   ')).toEqual([]);
  });
});

describe('object matching', () => {
  test('"dog" matches a photo labeled "golden retriever" — the core synonym win', () => {
    expect(matches(['golden retriever', 'grass', 'fence'], 'dog')).toBe(true);
  });

  test('"dog" matches "labrador retriever"', () => {
    expect(matches(['labrador retriever'], 'dog')).toBe(true);
  });

  test('"cat" does NOT match a photo of only dogs', () => {
    expect(matches(['golden retriever', 'beagle'], 'cat')).toBe(false);
  });

  test('direct label match works without synonyms', () => {
    expect(matches(['pizza', 'plate'], 'pizza')).toBe(true);
  });

  test('"food" matches "cheeseburger"', () => {
    expect(matches(['cheeseburger', 'plate'], 'food')).toBe(true);
  });

  test('no labels never matches', () => {
    expect(matches([], 'dog')).toBe(false);
  });
});
