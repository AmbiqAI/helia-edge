# heliaEDGE site

Astro/Starlight documentation using the pinned shared HELIA components. Tracks [#24](https://github.com/AmbiqAI/helia-edge/issues/24).

## Develop

Use Node 24, npm 11, Python 3.12+ and uv:

```sh
npm ci
npm run dev -- --port 8773
```

The site is served under `/helia-edge/`. The GitHub Pages workflow builds and verifies this site before deploying from `main`.

## Content and API

- `src/content/docs/` contains the migrated authored guides. Preserve backend support boundaries while the runtime API is being revised.
- `scripts/build-reference.mjs` reads `../helia_edge` statically with Griffe 1.7.3, including lazy `.pyi` exports. It generates API pages, a categorized search index, source links, and JSON/Markdown renditions using `helia-ui-pyref`. It does not install or import training frameworks.
- `scripts/build-notebooks.py` renders the committed notebooks as code, prose and saved output. It also publishes notebook downloads. It never executes training.
- Both scripts run before dev, build and check. Generated files are ignored; edit the source docstrings, notebooks or owning scripts instead.
- API categories come from module paths. No inferred backend support badges are assigned. Add backend filters only once the runtime provides a maintained support contract.
- `src/data/redirects.json` maps the old guide routes. Old API routes are mapped from the generated module inventory.
- Reserved namespaces retain routes for continuity but are omitted from navigation and the API catalog. Their source docstrings state their scope.
- `api-enrichment.mjs` resolves public import aliases and links internal inherited methods to their owners. It copies constructor metadata for single inheritance only; external bases are not expanded.
- `.cache/api-coverage.json` reports missing descriptions and package exports. Output validation rejects blank indexed descriptions; Python syntax is validated before extraction.

## Verify

```sh
npm run check
npm run build
npm run check:output
npx playwright install chromium
npm test
```

The output check follows local links and symbol anchors, checks representative public exports and ensures search and machine-readable artifacts exist. Browser tests cover filters, search, redirects, and narrow/wide layouts in light and dark themes.

`.github/workflows/docs.yaml` checks pull requests and uploads a preview artifact. Pushes to `main` and manual runs on `main` deploy only after the same checks pass. The Python backend matrix executes the first-model walkthrough and landing preprocessing example.

The shared UI package is pinned to a public Git commit over HTTPS. No SSH key or cross-repository secret is needed.

Design mockups under `src/mockups/` are injected only by the dev server at `/helia-edge/mockups/`; they are excluded from production output. Only the notebook sources remain in `docs/guides/`; edit authored site pages in `src/content/docs/`.
