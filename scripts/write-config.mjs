import { dirname } from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';

const values = {
  SUPABASE_URL: process.env.SUPABASE_URL || '',
  SUPABASE_ANON_KEY: process.env.SUPABASE_ANON_KEY || '',
  BREVO_HOST: process.env.BREVO_HOST || 'smtp-relay.brevo.com',
  BREVO_USER: process.env.BREVO_USER || '',
  BREVO_SMTP_PASS: process.env.BREVO_SMTP_PASS || '',
  BREVO_API_KEY: process.env.BREVO_API_KEY || '',
  BREVO_SENDER_EMAIL: process.env.BREVO_SENDER_EMAIL || '',
  BREVO_SENDER_NAME: process.env.BREVO_SENDER_NAME || 'TALENT.PREMIUM',
};

const config = Object.entries(values)
  .map(([key, value]) => `const ${key} = ${JSON.stringify(value)};`)
  .join('\n');

const target = process.argv[2] || 'config.js';
mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, `${config}\n`);
