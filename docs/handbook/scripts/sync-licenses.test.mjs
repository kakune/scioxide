import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import test from 'node:test';
import { syncSciOxideLicenses } from './sync-licenses.mjs';

const root = fileURLToPath(new URL('../../../', import.meta.url));

test('Wasm distribution carries the complete original and third-party notices', () => {
  const outputs = syncSciOxideLicenses();
  for (const name of ['LICENSE-MIT', 'LICENSE-APACHE']) {
    assert.equal(outputs[name + '.txt'], readFileSync(path.join(root, name), 'utf8'));
    assert.ok(outputs['NOTICE.txt'].includes(outputs[name + '.txt']));
  }
  for (const crate of readdirSync(path.join(root, 'crates'))) {
    const files = readdirSync(path.join(root, 'crates', crate));
    if (files.includes('THIRD_PARTY_NOTICES.md')) {
      const notice = readFileSync(path.join(root, 'crates', crate, 'THIRD_PARTY_NOTICES.md'), 'utf8');
      assert.ok(outputs['NOTICE.txt'].includes(notice), `${crate}: full notice missing`);
    }
  }
  assert.ok(outputs['NOTICE.txt'].includes('Cephes Math Library Release 2.8'));
  assert.ok(outputs['NOTICE.txt'].includes('Frances Y. Kuo and Stephen Joe'));
  assert.ok(outputs['NOTICE.txt'].includes('Robert Parrish'));
  assert.ok(outputs['NOTICE.txt'].includes('Copyright (c) 2004, Ernst Hairer'));
  const original = readFileSync(path.join(root, 'DATA_PROVENANCE.md'), 'utf8');
  assert.ok(outputs['NOTICE.txt'].includes(original));
});
