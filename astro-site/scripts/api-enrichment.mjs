export function enrichApi(root) {
  const nodes = new Map();
  function visit(node) {
    nodes.set(node.path, node);
    Object.values(node.members ?? {}).forEach(visit);
  }
  visit(root);
  function resolve(path, seen = new Set()) {
    if (seen.has(path)) return undefined;
    seen.add(path);
    const node = nodes.get(path);
    return node?.kind === 'alias' ? resolve(node.target_path, seen) : node;
  }
  const exports = new Map();
  for (const node of nodes.values()) {
    if (node.kind !== 'module' || !/\/__init__\.pyi?$/.test(node.filepath ?? '')) continue;
    for (const member of Object.values(node.members ?? {})) {
      if (member.name.startsWith('_')) continue;
      const target = resolve(member.path);
      if (!target || !['class', 'function'].includes(target.kind)) continue;
      const paths = exports.get(target.path) ?? [];
      paths.push(member.path);
      exports.set(target.path, paths);
    }
  }
  function baseClass(cls, base) {
    const module = nodes.get(cls.path.slice(0, cls.path.lastIndexOf('.')));
    const name = typeof base === 'string' ? base : base.name;
    return resolve(module?.members?.[name]?.path ?? name);
  }
  const link = (node) => {
    const module = [...nodes.values()].find(n => n.kind === 'module' && node.path.startsWith(n.path + '.') && n.members?.[node.name]?.path === node.path);
    return module ? `/helia-edge/reference/api/${module.path.replaceAll('.', '/').toLowerCase()}/#${node.path}` : undefined;
  };
  let inheritedClasses = 0;
  for (const cls of nodes.values()) {
    if (cls.kind !== 'class') continue;
    const bases = (cls.bases ?? []).map(base => baseClass(cls, base)).filter(base => base?.kind === 'class');
    if (!bases.length) continue;
    const lines = bases.map(base => `Base class: [${base.name}](${link(base)}).`);
    // Single inheritance has an unambiguous owner; multiple inheritance stays linked to its bases.
    if ((cls.bases ?? []).length === 1) {
      const owned = new Set(Object.keys(cls.members ?? {}));
      const seen = new Set([cls.path]);
      let base = bases[0];
      while (base && !seen.has(base.path)) {
        seen.add(base.path);
        if (!cls.members.__init__ && base.members?.__init__) {
          cls.members.__init__ = structuredClone(base.members.__init__);
        }
        const methods = Object.values(base.members ?? {}).filter(member => member.kind === 'function' && !member.name.startsWith('_') && !owned.has(member.name));
        if (methods.length) lines.push(`Inherited from [${base.name}](${link(base)}): ${methods.map(method => `[${method.name}()](${link(base)}.${method.name})`).join(', ')}.`);
        Object.keys(base.members ?? {}).forEach(name => owned.add(name));
        base = base.bases?.length === 1 ? baseClass(base, base.bases[0]) : undefined;
      }
    }
    const text = '\n\n' + lines.join('\n\n');
    cls.docstring ??= {value: cls.members.__init__?.docstring?.value ?? '', parsed: structuredClone((cls.members.__init__?.docstring?.parsed ?? []).filter(section => section.kind === 'text'))};
    cls.docstring.value += text;
    cls.docstring.parsed ??= [];
    cls.docstring.parsed.push({kind: 'text', value: text});
    inheritedClasses++;
  }
  return {exports, inheritedClasses};
}
