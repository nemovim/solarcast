// Load one root .env for local development and pass values to Next.js workers.
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const frontend = path.resolve(__dirname, '..');
const root = path.resolve(frontend, '..');
const { loadEnvConfig } = require('@next/env');
const command = process.argv[2] || 'dev';
if (!['dev', 'build', 'start'].includes(command)) {
  throw new Error('Usage: npm run dev|build|start');
}
loadEnvConfig(root, command === 'dev');
const args = [path.join(frontend, 'node_modules/next/dist/bin/next'), command];
if (command === 'dev') args.push('--hostname', '0.0.0.0', '--port', '5173');
args.push(...process.argv.slice(3));
const result = spawnSync(process.execPath, args, { cwd: frontend, env: process.env, stdio: 'inherit' });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
