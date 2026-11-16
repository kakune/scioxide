import { readFileSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const project = fileURLToPath(new URL('../', import.meta.url));
const root = path.resolve(project, '../..');

// This supplements Oxiquill's notices, which do not cover consumer helper crates.
export function syncSciOxideLicenses() {
  const manifest = readFileSync(path.join(root, 'Cargo.toml'), 'utf8');
  const members = [...manifest.match(/members\s*=\s*\[([\s\S]*?)\]/)[1].matchAll(/"(crates\/[^\"]+)"/g)]
    .map((match) => match[1]);
  const source = (relative) => readFileSync(path.join(root, relative), 'utf8');
  const parts = [
    'SciOxide copyright, licenses and third-party redistribution notices\n',
    source('LICENSE-MIT'), source('LICENSE-APACHE'), source('DATA_PROVENANCE.md')
  ];
  for (const member of members) {
    const notice = path.join(root, member, 'THIRD_PARTY_NOTICES.md');
    if (existsSync(notice)) {
      parts.push(`\n${'='.repeat(80)}\n${path.basename(member)}\n${'='.repeat(80)}\n`);
      parts.push(readFileSync(notice, 'utf8'));
    }
  }
  const outputs = {
    'NOTICE.txt': parts.join('\n'),
    'LICENSE-MIT.txt': source('LICENSE-MIT'),
    'LICENSE-APACHE.txt': source('LICENSE-APACHE')
  };
  const directory = path.join(project, 'public/scioxide-licenses');
  mkdirSync(directory, { recursive: true });
  for (const [name, text] of Object.entries(outputs)) {
    const dest = path.join(directory, name);
    if (!existsSync(dest) || readFileSync(dest, 'utf8') !== text) writeFileSync(dest, text);
  }
  return outputs;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  syncSciOxideLicenses();
}
