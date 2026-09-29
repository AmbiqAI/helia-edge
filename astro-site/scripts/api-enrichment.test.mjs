import { test } from 'node:test';
import assert from 'node:assert/strict';
import { enrichApi } from './api-enrichment.mjs';

const fn = (path) => ({kind:'function',path,name:path.split('.').at(-1),parameters:[{name:'model'}]});
test('lazy public exports and inherited contracts retain their defining owner', () => {
  const parent = {kind:'class',path:'helia_edge.converter.Parent',name:'Parent',members:{__init__:fn('helia_edge.converter.Parent.__init__'),export:fn('helia_edge.converter.Parent.export'),convert:fn('helia_edge.converter.Parent.convert')}};
  const child = {kind:'class',path:'helia_edge.lite.Child',name:'Child',bases:[{name:'Parent'}],members:{convert:fn('helia_edge.lite.Child.convert')}};
  const root = {kind:'module',path:'helia_edge',filepath:'/src/helia_edge/__init__.py',members:{
    Child:{kind:'alias',name:'Child',path:'helia_edge.Child',target_path:child.path},
    converter:{name:'converter',kind:'module',path:'helia_edge.converter',members:{Parent:parent}},
    lite:{name:'lite',kind:'module',path:'helia_edge.lite',members:{Parent:{kind:'alias',path:'helia_edge.lite.Parent',target_path:parent.path},Child:child}},
  }};
  const result = enrichApi(root);
  assert.deepEqual(result.exports.get(child.path),['helia_edge.Child']);
  assert.deepEqual(child.members.__init__.parameters,[{name:'model'}]);
  assert.equal(child.members.__init__.path,parent.path+'.__init__');
  assert.match(child.docstring.value,/#helia_edge.converter.Parent.export/);
  assert.doesNotMatch(child.docstring.value,/Parent.convert/);
  assert.equal(parent.docstring,undefined);
});

test('external bases do not invent method resolution', () => {
  const cls = {kind:'class',path:'helia_edge.A',name:'A',bases:[{name:'keras.Model'}],members:{}};
  const root={kind:'module',path:'helia_edge',members:{A:cls}};
  enrichApi(root);
  assert.equal(cls.members.__init__,undefined);
  assert.equal(cls.docstring,undefined);
});
