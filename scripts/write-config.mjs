import { dirname } from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';

const values = {
  SUPABASE_URL: process.env.SUPABASE_URL || '',
  SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY || '',
};

const config = Object.entries(values)
  .map(([key, value]) => `const ${key} = ${JSON.stringify(value)};`)
  .join('\n');

const target = process.argv[2] || 'config.js';
mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, `${config}\n`);
