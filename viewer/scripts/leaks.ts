/**
 * What must never be published: a home folder path, a model provider's endpoint address, or a
 * `_`-prefixed field. URLs in general are allowed, since model replies can name help lines.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const HOME_PATH = /(?:\/Users\/|\/home\/|[A-Za-z]:\\Users\\)[^\s"'\\/]+/;

const ENDPOINT =
  /https?:\/\/[^\s"'<>]*?(?:openai\.azure\.com|cognitiveservices\.azure\.com|azure-api\.net|api\.openai\.com|api\.anthropic\.com|amazonaws\.com|googleapis\.com|localhost|127\.0\.0\.1|0\.0\.0\.0)/i;

export interface Leak {
  file: string;
  kind: 'home path' | 'endpoint address' | 'internal field';
  sample: string;
}

function internalKeys(value: unknown, path = ''): string[] {
  if (Array.isArray(value)) return value.flatMap((item, i) => internalKeys(item, `${path}[${i}]`));
  if (value === null || typeof value !== 'object') return [];
  return Object.entries(value).flatMap(([key, item]) => [
    ...(key.startsWith('_') ? [`${path}.${key}`] : []),
    ...internalKeys(item, `${path}.${key}`),
  ]);
}

/** Leaks in one file's text. JSON and JSON Lines are also checked for `_`-prefixed keys. */
export function leaksIn(file: string, text: string): Leak[] {
  const found: Leak[] = [];
  const home = HOME_PATH.exec(text);
  if (home) found.push({ file, kind: 'home path', sample: home[0] });
  const endpoint = ENDPOINT.exec(text);
  if (endpoint) found.push({ file, kind: 'endpoint address', sample: endpoint[0] });
  const docs = file.endsWith('.jsonl') ? text.split('\n') : file.endsWith('.json') ? [text] : [];
  for (const doc of docs) {
    if (!doc.trim()) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(doc);
    } catch {
      continue;
    }
    const keys = internalKeys(parsed);
    if (keys.length) {
      found.push({ file, kind: 'internal field', sample: keys[0]! });
      break;
    }
  }
  return found;
}

export function filesUnder(root: string): string[] {
  let names: string[];
  try {
    names = readdirSync(root, { recursive: true, encoding: 'utf-8' });
  } catch {
    return [];
  }
  return names.map((name) => join(root, name)).filter((path) => statSync(path).isFile());
}

export function scan(roots: string[], base: string): Leak[] {
  return roots.flatMap((root) =>
    filesUnder(root).flatMap((path) => leaksIn(relative(base, path), readFileSync(path, 'utf-8'))),
  );
}
