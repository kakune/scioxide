import assert from 'node:assert/strict';
import test from 'node:test';
import { chapterSidebar } from './chapter-sidebar.mjs';

test('preserves current chapter, translated overview URLs and source navigation', () => {
  const link = (href, isCurrent = false) => ({
    type: 'link', label: 'Overview', href, isCurrent, attrs: {}, badge: undefined
  });
  const group = (label, entries) => ({ type: 'group', label, entries, collapsed: true, badge: undefined });
  const current = group('Core', [
    link('/scioxide/ja/scioxide-core/'),
    group('API', [link('/scioxide/ja/scioxide-core/api/', true)])
  ]);
  const inactive = group('多項式', [
    group('Methods', [link('/scioxide/ja/scioxide-poly/'), link('/scioxide/ja/scioxide-poly/api/')])
  ]);
  const sidebar = [current, inactive, group('Missing translation', [])];
  const snapshot = structuredClone(sidebar);
  const result = chapterSidebar(sidebar);
  assert.equal(result.length, 2);
  assert.equal(result[0], current);
  assert.equal(result[1].type, 'link');
  assert.equal(result[1].label, '多項式');
  assert.equal(result[1].href, '/scioxide/ja/scioxide-poly/');
  assert.deepEqual(sidebar, snapshot);
});
