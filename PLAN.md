# CommonCatalog SFT ImageDataset Builder

## Summary

Build a resumable Python/`uv` pipeline that creates an ImageDataset-style output from `common-canvas/commoncatalog-cc-by`: about 200 GB of downloaded image files, `>=1024x1024`, near-square ratio `0.9..1.1`, max `3072 px` on the larger side or `20 MB` by `Content-Length`, with `blip2_caption`, all original metadata except `caption`, quality metrics, and a `quality_label`.

## Key Changes

- Add a `uv` Python CLI with commands:
  - `prepare-sft-dataset run --config config.yaml`
  - `prepare-sft-dataset inspect-commoncatalog --limit-shards N`
  - `prepare-sft-dataset validate-output --dataset-root PATH`
- Stream parquet shards from `hf://datasets/common-canvas/commoncatalog-cc-by/...`; do not download the full dataset.
- Walk source shards deterministically: top dirs `0..9`, least-dim buckets `1024-2048`, `2048-4096`, `4096+`, and near-square aspect buckets.
- Filter before downloading: dimensions `>=1024`, ratio `0.9..1.1`, larger side `<=3072`, non-empty `blip2_caption`, usable URL.
- Use configurable async download concurrency, default `64`, with HEAD/body size caps and atomic file writes.
- Stop when saved image bytes reach about `200 GiB`.

## Filtering And Labels

- Preserve original downloaded image bytes.
- Store files under `images/good`, `images/bad`, and `images/bad/synthetic`.
- Score images with decode, dimensions, ratio, file size, luma/color variation, entropy, blur, and edge density metrics.
- Mark failed, monotone, blurry, truncated, unsupported, or out-of-policy images as `bad` instead of deleting.
- Maintain bad images at target `7.5%` of sample count, hard cap `10%`; generated synthetic bad rows are pinned.
- Generate solid color, white-noise, and low-contrast-noise synthetic bad examples before normal processing.

## Output Format

- Output root contains `images`, `metadata`, `state`, and `reports`.
- Metadata parquet rows contain relative path, `blip2_caption`, quality labels, all original CommonCatalog metadata except `caption`, quality metrics, and source bookkeeping.
- Flush metadata every `2048` accepted rows or `2 GiB` saved image bytes.

## Resumability

- Downloads write to `.tmp`, verify, then atomically rename.
- `state/state.json` tracks current shard/row, output part, saved bytes, counts, RNG seed, and worker config.
- `state/seen_ids.sqlite` prevents duplicates across restarts.
- Startup removes dangling temp files and resumes from committed state.

## Test Plan

- Unit tests cover bucket parsing, metadata filtering, max-size filtering, schema behavior, synthetic generation, quality scoring, and reservoir cap behavior.
- Integration tests use tiny local parquet fixtures and local HTTP image serving.
- Dry-run mode reports candidate shards, config, and schema without downloading images.

## Assumptions

- Bad image ratio is by sample count, target `7.5%`, hard cap `10%`.
- Too large means larger side over `3072 px` by metadata or body over `20 MB`.
- Download throttling uses configurable concurrency only.
- Resizing to `512` happens later during training/preprocessing.
