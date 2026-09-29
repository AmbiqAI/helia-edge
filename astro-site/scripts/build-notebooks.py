"""Render committed notebook cells without executing training code."""
import base64
import json
from pathlib import Path
import re
import shutil

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
for source in sorted((REPO / 'docs/guides').glob('*.ipynb')):
    notebook = json.loads(source.read_text())
    title = {'train-cifar-model': 'Train a CIFAR-10 model', 'custom-model-architecture': 'Create a custom architecture'}[source.stem]
    parts = [f'---\ntitle: {title}\ndescription: A heliaEDGE notebook with source code and saved outputs.\n---\n',
             f'[Download the notebook](/helia-edge/notebooks/{source.name})\n\nThis page includes the notebook’s saved outputs. It does not run training in your browser.\n']
    downloads = SITE / 'public/notebooks'
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, downloads / source.name)
    for ci, cell in enumerate(notebook['cells']):
        text = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            if '<div class="grid cards"' in text:
                parts.append(f'[Open in Colab](https://colab.research.google.com/github/AmbiqAI/helia-edge/blob/main/docs/guides/{source.name}) · [View source](https://github.com/AmbiqAI/helia-edge/blob/main/docs/guides/{source.name})')
                continue
            text = re.sub(r':[a-z]+-[a-z0-9-]+:', '', text)
            if text.startswith('# '):
                text = '#' + text
            parts.append(text)
        elif cell['cell_type'] == 'code':
            parts.append('```python\n' + text.rstrip() + '\n```')
            output = []
            for oi, item in enumerate(cell.get('outputs', [])):
                data = item.get('data', {})
                if 'image/png' in data:
                    name = f'{source.stem}-{ci}-{oi}.png'
                    (downloads / name).write_bytes(base64.b64decode(''.join(data['image/png'])))
                    output.append(f'![Saved notebook figure](/helia-edge/notebooks/{name})')
                else:
                    raw = ''.join(item.get('text', data.get('text/plain', [])))
                    raw = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw)
                    if raw.strip():
                        output.append('```text\n' + raw.rstrip() + '\n```')
            if output:
                parts.append('<details>\n<summary>Saved output</summary>\n\n' + '\n\n'.join(output) + '\n\n</details>')
    dest = SITE / 'src/content/docs/examples' / f'{source.stem}.md'
    dest.write_text('\n\n'.join(parts) + '\n')
