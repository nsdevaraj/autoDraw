# AutoDraw Offline

This repository contains a local AutoDraw-style canvas, the browser retrieval runtime, SVG suggestions, curation tooling, and the reproducible model pipeline.

Suggestions are produced by embedding the sketch and searching an index that covers **every icon in the pinned SVGDepot commit** (210,199 files, 185,409 distinct shapes). The ONNX model is a shared encoder, not a per-icon classifier: it maps a 64x64 stroke bitmap to a 64-dimension unit vector, so corpus coverage is a property of the index rather than of the model weights. Icons outside the 37 supervised classes are reached zero-shot on shape.

## Requirements

- Node.js 24 and npm
- Python 3.9 only if you want to rebuild or fully verify the model

## Run AutoDraw Local

Install the JavaScript dependencies:

```sh
npm ci
```

Start the local server:

```sh
node scripts/serve-curation.mjs
```

Open <http://127.0.0.1:4173/>. Stop the server with `Ctrl+C`.

To use a different address or port:

```sh
node scripts/serve-curation.mjs --host 0.0.0.0 --port 8080
```

Draw on the canvas and choose any suggestion to replace the active sketch. All ranked candidates are treated as approved; curation is not required. The editor includes undo, redo, clear, ink controls, and SVG export.

Icon URLs are never stored in the index. They are rebuilt from the pinned commit through `expectedIconUrl()`, so the approved-URL invariant holds by construction. The app requests icons through a local route that serves `.cache/svgdepot` when present and otherwise proxies the commit-pinned jsDelivr URL, so a clean clone runs without preprocessing.

If the embedder or the icon index cannot be loaded, the editor falls back to the classifier and the approved candidate manifest, which only covers the 37 supervised classes.

The optional curator remains available at <http://127.0.0.1:4173/curate.html>.

## Deploy to Vercel

The Vercel build copies only the browser application, candidate manifest, models, icon embedding index, and ONNX Runtime files into `dist/`. Missing local SVGDepot icons are served through a Vercel Function that accepts only confined SVG paths under the pinned commit.

Build the static output locally:

```sh
node scripts/build-vercel.mjs
```

Deploy a preview directly from this directory:

```sh
npx vercel
```

After verifying the preview, deploy it to production:

```sh
npx vercel --prod
```

No environment variables are required. For automatic deployments, push this repository to a Git provider and import it in Vercel with the **Other** framework preset. The build command, output directory, Function files, rewrite, MIME types, and security headers are defined in `vercel.json`.

## Run the tests

```sh
npm test
```

This checks the curation code, rasterizer, retrieval runtime, training pipeline, tracked ONNX models, icon embedding index, and ONNX Runtime Web integration. On a clean clone, the expensive cache-backed metric recomputation is skipped because the training cache is not tracked.

## Rebuild the icon embedding index

The tracked index under `data/icon-embeddings/` is ready to use. Rebuilding it requires the SVG bytes for the whole pinned commit.

Ingest the corpus. A local checkout is far faster than 210k CDN requests, and the commit is verified against the index before anything is read:

```sh
node scripts/fetch-svgdepot-icons.mjs --source-root ../SVGDepot
```

Without `--source-root` the same command downloads from the commit-pinned jsDelivr CDN into a resumable, content-addressed store.

Then encode, cluster, and shard:

```sh
.venv-model/bin/python scripts/build-icon-embedding-index.py --jobs 8
```

This rasterizes every distinct SVG through the same worker the training cache uses, encodes it with the tracked embedder, deduplicates identical bitmaps, runs a deterministic spherical k-means (k=256), and writes `index.json` plus one `.bin` and one `.meta.json.gz` per shard. First paint fetches `index.json` and only the probed shards.

## Rebuild and verify the model

The committed ONNX model is ready to use. Rebuilding it is optional and downloads the trusted Sketch-RNN dataset locally.

```sh
python3 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-model.txt

python scripts/fetch-sketchrnn-data.py --jobs 4
python scripts/build-training-cache.py
python scripts/train-quickdraw-model.py
python scripts/verify-quickdraw-model.py
```

To require the full cache-backed verification in the Node test suite:

```sh
AUTODRAW_REQUIRE_MODEL_CACHE=1 node --test tests/model-artifact.test.mjs
```

Downloaded archives and generated training caches are intentionally excluded from Git.
