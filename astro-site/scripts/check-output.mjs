import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { resolve, relative, join } from 'node:path';
import { parseHTML } from 'linkedom';
import assert from 'node:assert/strict';
const root = resolve('dist');
const files = (dir) => readdirSync(dir).flatMap(n => statSync(join(dir,n)).isDirectory() ? files(join(dir,n)) : [join(dir,n)]);
const errors = [];
const cache = new Map();
const doc = (file) => {
  if (!cache.has(file)) cache.set(file,parseHTML(readFileSync(file,'utf8')).document);
  return cache.get(file);
};
for (const file of files(root).filter(f=>f.endsWith('.html'))) {
  const from = new URL('/helia-edge/'+relative(root,file).replace(/index\.html$/, ''),'https://ambiqai.github.io');
  for (const a of doc(file).querySelectorAll('a[href], img[src]')) {
    const raw = a.getAttribute('href') ?? a.getAttribute('src');
    if (!raw || raw.startsWith('mailto:')) continue;
    const url = new URL(raw,from);
    if (url.origin !== from.origin || !url.pathname.startsWith('/helia-edge/')) continue;
    const path = resolve(root,decodeURIComponent(url.pathname.slice('/helia-edge/'.length)));
    const target = [path,join(path,'index.html'),path+'.html'].find(p=>existsSync(p)&&statSync(p).isFile());
    if (!target) errors.push(`${relative(root,file)} -> ${raw}`);
    else if (url.hash && target.endsWith('.html') && !doc(target).getElementById(decodeURIComponent(url.hash.slice(1)))) errors.push(`${relative(root,file)} -> missing anchor ${raw}`);
  }
}
const index=JSON.parse(readFileSync('src/data/api-index.json','utf8'));
assert(index.rows.length > 200, 'API index unexpectedly lost coverage');
for (const name of ['TcnModel','RandomGaussianNoise1D','TQDMProgressBar','register_keras_serializables']) assert(index.rows.some(r=>r.name===name), `Missing public API: ${name}`);
assert(readFileSync('dist/reference/api/helia_edge/models/tcn/index.md','utf8').includes('TcnParams'), 'Semantic API Markdown missing');
assert(existsSync('dist/llms-full.txt'), 'Missing LLM export');
assert(existsSync('dist/pagefind/pagefind.js'), 'Missing search index');
if(errors.length) throw Error(`Broken internal links (${errors.length}):\n${[...new Set(errors)].join('\n')}`);
console.log(`Checked internal links and anchors across ${cache.size} HTML pages; ${index.rows.length} API entries and discovery artifacts.`);

const coverage=JSON.parse(readFileSync('.cache/api-coverage.json','utf8'));
assert.deepEqual(coverage.missingDescriptions, [], 'Public API descriptions must be documented in source');
const litert=readFileSync('dist/reference/api/helia_edge/converters/litert/converter/index.md','utf8');
assert(litert.includes('TfLiteKerasConverter.export_header'), 'Inherited converter export contract missing');
assert(litert.includes('model'), 'Inherited constructor contract missing');
const exported=index.rows.find(row=>row.name==='ConfusionMatrix');
assert(exported.publicPaths.includes('helia_edge.metrics.ConfusionMatrix'), 'Canonical metrics import missing');

assert(readFileSync("dist/index.md","utf8").includes("metric.update_state"), "Landing workbench missing from Markdown export");

assert(!existsSync("dist/mockups"), "Design mockups must not ship in production");
