import {readFileSync,writeFileSync} from 'node:fs';
const steps=JSON.parse(readFileSync('src/data/workbench.json','utf8'));
const intro='Start with your task or explore a stage. Each example connects to a practical guide.';
const text=steps.map(step=>`### ${step.label}: ${step.title}\n\n${step.text}\n\n\`\`\`python\n${step.code}\n\`\`\`\n\n${step.note}\n\n[${step.label} guide](https://ambiqai.github.io/helia-edge/guide/${step.guide}/)`).join('\n\n');
for(const path of ['dist/index.md','dist/llms-full.txt']){
 const source=readFileSync(path,'utf8');
 if(!source.includes(intro))throw Error(`Landing export anchor missing: ${path}`);
 writeFileSync(path,source.replace(intro,`${intro}\n\n${text}`));
}
