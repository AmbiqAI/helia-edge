import { enrichApi } from './api-enrichment.mjs';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = fileURLToPath(new URL('../', import.meta.url));
const repo = resolve(site, '..');
const cache = resolve(site, '.cache');
mkdirSync(cache, { recursive: true });
const run = (command, args) => execFileSync(command, args, { cwd: repo, encoding: 'utf8', maxBuffer: 128 * 1024 * 1024 });
const commit = run('git', ['rev-parse', 'HEAD']).trim();
run('uv', ['run', '--no-project', '--python', '3.12', 'python', '-c', "import ast,pathlib; [ast.parse(p.read_text(), filename=str(p)) for p in pathlib.Path('helia_edge').rglob('*.py')]"]);
const dump = JSON.parse(run('uv', ['tool', 'run', '--from', 'griffe==1.7.3', 'griffe', 'dump', 'helia_edge', '--docstyle', 'google', '-f']));
// Lazy exports are declared in .pyi files. Griffe merges those without importing either backend.
function scope(node) {
  if (node.docstring) {
    node.docstring.value = node.docstring.value.replace(/:(?:material|simple)-[a-z0-9-]+:/g, "");
    for (const section of node.docstring.parsed ?? []) {
      if (section.kind === "text") section.value = section.value.replace(/:(?:material|simple)-[a-z0-9-]+:/g, "");
    }
  }
  for (const [name, member] of Object.entries(node.members ?? {})) {
    if (member.kind === 'module') {
      if (name.startsWith('_')) delete node.members[name];
      else scope(member);
    } else if (member.kind === 'alias' && !member.target_path?.startsWith('helia_edge.')) {
      delete node.members[name];
    } else if (['class', 'function'].includes(member.kind)) scope(member);
  }
}
// Public exports may be implemented in private modules that have no reference page.
function resolveAlias(path, seen = new Set()) {
  if (seen.has(path)) throw new Error(`Cyclic public export: ${path}`);
  seen.add(path);
  const parts = path.split('.');
  let member = dump[parts.shift()];
  for (const part of parts) member = member?.members?.[part];
  if (!member) throw new Error(`Unresolved public export: ${path}`);
  return member.kind === 'alias' ? resolveAlias(member.target_path, seen) : member;
}
function materializePrivateExports(node) {
  for (const [name, member] of Object.entries(node.members ?? {})) {
    if (name.startsWith('_')) continue;
    if (member.kind === 'module') materializePrivateExports(member);
    else if (member.kind === 'alias' && member.target_path?.startsWith('helia_edge.')) {
      const targetParts = member.target_path.split('.').slice(1,-1);
      if (node.filepath?.endsWith('/__init__.py') && targetParts.some(part => part.startsWith('_'))) {
        node.members[name] = { ...structuredClone(resolveAlias(member.target_path)), name, path: member.path };
      }
    }
  }
}
materializePrivateExports(dump.helia_edge);
scope(dump.helia_edge);
const enrichment = enrichApi(dump.helia_edge);
writeFileSync(resolve(cache, 'griffe.json'), JSON.stringify(dump));
run(process.execPath, [resolve(site, 'node_modules/@ambiqai/helia-ui/scripts/pyref.mjs'),
  '--input', resolve(cache, 'griffe.json'), '--out', resolve(site, 'src/content/docs/reference/api'),
  '--public', resolve(site, 'public'), '--base', '/helia-edge/', '--site', 'https://ambiqai.github.io',
  '--source-root', repo, '--source-url', `https://github.com/AmbiqAI/helia-edge/blob/${commit}/{path}#L{line}`,
  '--commit', commit, '--quiet']);
const model = JSON.parse(readFileSync(resolve(site, 'public/reference/api/reference.json'), 'utf8'));
const modules = [];
const visit = (m) => { modules.push(m); (m.submodules ?? []).forEach(visit); };
model.modules.forEach(visit);
const slug = (s) => s.toLowerCase();
const route = (m) => `reference/api/${m.path.split('.').map(slug).join('/')}`;
const groups = {models:'Architectures', callbacks:'Callbacks', export:'Conversion', data:'Data', losses:'Losses', metrics:'Metrics', plotting:'Plotting', trainers:'Training', utils:'Utilities', layers:'Layers'};
function category(m) {
  if (m.path.startsWith('helia_edge.layers.preprocessing')) {
    return /random|augment|warp|mix_style|sine_wave/.test(m.path) ? 'Augmentation' : 'Preprocessing';
  }
  return groups[m.path.split('.')[1]] ?? 'Package';
}
const sidebar = [];
const rows = [];
for (const m of modules) {
  const group = category(m);
  let section = sidebar.find(s => s.label === group);
  if (!section) sidebar.push(section = {label:group, collapsed:true, items:[]});
  if (m.symbols?.length || m.submodules?.length) section.items.push({label:m.path.replace(/^helia_edge\.?/, '') || 'helia_edge', slug:route(m)});
  for (const s of m.symbols ?? []) {
    if (!['class','function'].includes(s.kind)) continue;
    const publicPaths = enrichment.exports.get(s.id) ?? [];
    rows.push({publicPaths, id:s.id, name:s.name, kind:s.kind, module:m.path, group,
      href:`/helia-edge/${route(m)}/#${s.id}`, summary:(s.description ?? '').trim().split('\n')[0], facets:{category:[group],kind:[s.kind],module:[m.path], exports:publicPaths, surface:[publicPaths.length ? 'Package export' : 'Module API']}});
  }
}
mkdirSync(resolve(site, 'src/data'), { recursive:true });
writeFileSync(resolve(site, 'src/data/api-redirects.json'), JSON.stringify(Object.fromEntries(modules.filter(m => !m.path.includes('._')).map(m => ['/api/'+m.path.replaceAll('.', '/'), '/helia-edge/'+route(m)+'/'])),null,2)+'\n');
const order = ['Architectures','Preprocessing','Augmentation','Layers','Metrics','Callbacks','Losses','Training','Conversion','Inference','Plotting','Utilities','Package'];
sidebar.sort((a,b)=>order.indexOf(a.label)-order.indexOf(b.label));
writeFileSync(resolve(site, 'src/data/api-sidebar.json'), JSON.stringify(sidebar,null,2)+'\n');
writeFileSync(resolve(site, 'src/data/api-index.json'), JSON.stringify({rows, filters:[{id:'surface',label:'Import surface',values:['Package export','Module API']},{id:'category',label:'Category',values:[...new Set(rows.map(r=>r.group))].sort()},{id:'kind',label:'Symbol type',values:['class','function']}]},null,2)+'\n');
console.log(`Generated ${modules.length} modules and ${rows.length} searchable API entries from ${commit.slice(0,8)}.`);

const coverage = {
  sourceCommit: commit,
  inheritedClasses: enrichment.inheritedClasses,
  packageExports: rows.filter(row => row.publicPaths.length).length,
  modules: modules.length,
  indexedSymbols: rows.length,
  missingDescriptions: rows.filter(row => !row.summary.trim()).map(row => row.id),
  emptyModules: modules.filter(m => !(m.symbols?.length || m.submodules?.length)).map(m => m.path),
};
writeFileSync(resolve(cache, 'api-coverage.json'), JSON.stringify(coverage,null,2)+'\n');
