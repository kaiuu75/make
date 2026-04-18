# Approach — osapiens Makeathon 2026 Deforestation Detection

## 1. Problem summary

Produce a single `FeatureCollection` GeoJSON in EPSG:4326 whose polygons cover
every pixel on the hidden test tiles that was **forest in 2020** and became
**not‑forest after 2020**. Scoring:

| Metric                  | Role                 |
| ----------------------- | -------------------- |
| **Union IoU**           | main leaderboard     |
| Polygon Recall          | sanity check         |
| Polygon Level FPR       | precision sanity     |
| Year Accuracy           | bonus (`time_step`)  |

A polygon is treated as "everything inside = deforestation". There is **no
partial‑polygon credit**, so keeping predicted polygons compact and
well‑separated matters more than raw pixel accuracy.

## 2. Is the paper's approach (U‑TAE on Sentinel‑1) the right fit?

Karaman et al. 2023 (ISPRS, *BraDD‑S1TS*) frame deforestation detection as
semantic segmentation over **Sentinel‑1 C‑band SAR time series** using
U‑TAE (U‑Net + Temporal Attention Encoder), trained on PRODES alerts in the
Brazilian Amazon.

**What transfers well**

- SAR is cloud‑independent → valuable in tropical test tiles.
- Multi‑temporal modeling > bi‑temporal change detection (their Fig. 6:
  3.3 % IoU with 1 date vs. ~47 % with full series).
- Focal loss for class imbalance is a reusable training trick.
- U‑Net style spatial decoder produces clean polygon boundaries, which is
  good for the polygon metrics.

**What does not transfer**

1. **Modality scope** — the paper uses only S1. This challenge ships
   S1 + S2 + **AlphaEarth Foundations 64‑dim annual embeddings**. AEF is a
   pretrained foundation model: it already encodes the spatio‑temporal
   structure U‑TAE learns from scratch. Ignoring AEF leaves a lot of signal
   on the table.
2. **Region scope** — PRODES covers Brazil only. The makeathon explicitly
   asks for generalization to Africa and beyond. A model trained only on
   Amazonian SAR statistics will not generalize to e.g. miombo woodland,
   West‑African cocoa, or Indonesian palm. AEF was trained globally and is
   a much more defensible backbone for cross‑region generalization.
3. **Label regime** — the paper uses one clean, expert‑annotated label
   source. The challenge provides three **weak, noisy, conflicting**
   sources (RADD, GLAD‑L, GLAD‑S2). Reliable training requires a label
   **fusion** step (weighted consensus), not present in the paper.
4. **Compute** — U‑TAE on multi‑year monthly S1+S2 stacks over many tiles
   is not practical on a MacBook without pre‑chunked patches. A gradient
   boosting classifier on AEF Δ‑vectors runs in minutes on CPU.
5. **Target metric** — the paper reports pixel IoU on 48×48 patches. Our
   target is **Union IoU of polygons across the whole test set**. Our
   pipeline must be polygon‑aware (morphology, area filter, merging
   across tiles) from the start.
6. **Temporal target** — the paper does **not** predict when the event
   occurred. The challenge rewards `time_step` (YYMM) via Year Accuracy.
   We need an additional date‑assignment mechanism.

**Verdict**

A pure replication of the paper would under‑use the provided data, would
not generalize across continents, and would be slow to iterate with on a
MacBook. We treat U‑TAE as a **Tier 2** option (used only to refine
`time_step`) and build the primary pipeline around **AEF + weak‑label
fusion + LightGBM**.

## 3. Our pipeline

```
                ┌────────────────────────┐
Sentinel-1 ────▶│ annual VV statistics   │
(monthly VV)    │ mean/std/min 2020+last │
                └───────────┬────────────┘
                            │
                ┌────────────▼───────────┐
Sentinel-2 ────▶│ annual NDVI/NDMI stats │                    ┌───────────────┐
(monthly 12b)   │ median, slope, min     ├───────────────────▶│               │
                └───────────┬────────────┘                    │   LightGBM    │
                            │                                 │   pixel       │
                ┌────────────▼───────────┐                    │   classifier  │
AlphaEarth ────▶│ AEF(2020), AEF(last),  │                    │               │
(annual 64d)    │ Δ = last − 2020        ├───────────────────▶│               │
                └────────────┬───────────┘                    └───────┬───────┘
                             │                                        │
                             ▼                                        │
                   ┌──────────────────┐                                │
RADD / GLAD-L ────▶│ weak-label       │───── training targets ─────────┘
GLAD-S2            │ fusion (weighted │
                   │ consensus)       │
                   └────────┬─────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │  2020 forest mask  │ ← AEF-2020 clustering / Hansen
                  └────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ per-pixel probability │
                 │ × forest_mask(2020)   │
                 └────────┬──────────────┘
                            │
              threshold + morphology + area ≥ 0.5 ha
                            │
                            ▼
                 ┌──────────────────────┐
                 │ polygons + time_step │ ─── submission.geojson
                 └──────────────────────┘
```

### 3.1 Tier 0 — Zero‑training consensus baseline

Purpose: a submission that is valid, non‑trivial, and produced in minutes
from weak labels alone. Gives a lower bound before any training.

- Parse each label source into `(mask, date)` pixel rasters.
- RADD: leading digit → confidence (2 low, 3 high); remaining digits → days
  since 2014‑12‑31. Keep alerts whose date ≥ 2020‑01‑01.
- GLAD‑L: `alertYY` (2 prob / 3 conf) + `alertDateYY` (DOY). Combine across
  years ≥ 2020.
- GLAD‑S2: `alert` (0–4) + `alertDate` (days since 2019‑01‑01). Keep dates
  ≥ 2020‑01‑01.
- Reproject all three to the common Sentinel‑2 UTM grid.
- Per‑pixel confidence = max over sources, with scores:
  `RADD_high=1.0, RADD_low=0.6, GLAD‑L_conf=0.9, GLAD‑L_prob=0.5,
   GLAD‑S2_4=1.0, GLAD‑S2_3=0.75, GLAD‑S2_2=0.5, GLAD‑S2_1=0.25`.
- Positive if confidence ≥ 0.7 **and** any two sources agree, **OR** a
  single source is high‑confidence.
- Multiply by the 2020 forest mask.
- Morphological opening (3 px) + closing (5 px), drop < 0.5 ha,
  polygonize.
- `time_step` per polygon = YYMM of the median alert date of inside pixels.

### 3.2 Tier 1 — LightGBM pixel classifier (main submission)

Why LightGBM

- Handles noisy labels well (robust to class‑label flips).
- Operates on a per‑pixel feature vector — no patch loader, no GPU,
  trains in minutes on a laptop.
- Global features: AEF embeddings trained on the whole planet → inherent
  cross‑region generalization.

**Features per pixel**

| Group | Feature                                    | Dim |
|-------|--------------------------------------------|-----|
| AEF   | AEF(2020)                                  | 64  |
| AEF   | AEF(year_last)                             | 64  |
| AEF   | AEF(year_last) − AEF(2020)                 | 64  |
| S1    | mean/std/min VV for 2020                   | 3   |
| S1    | mean/std/min VV for year_last              | 3   |
| S1    | Δ mean VV, Δ std VV                        | 2   |
| S2    | median NDVI, median NDMI (2020, year_last) | 4   |
| S2    | Δ median NDVI, Δ median NDMI               | 2   |
| S2    | NDVI trend slope over all months ≥ 2020    | 1   |
| Total |                                            | 207 |

**Target** — the fused weak label (0/1) from §3.1.
**Sample weight** — the continuous confidence score from §3.1 on
positives, 1.0 on confirmed negatives, 0 on ambiguous pixels (ignored).

**Training**

- `objective=binary`, `boosting_type=gbdt`, `num_leaves=63`,
  `learning_rate=0.05`, ~500 rounds with early stopping on held‑out
  tiles (by region, to measure cross‑region generalization).
- Subsample pixels stratified on label and tile (e.g. 200k positives +
  400k negatives total).

**Inference** — per‑tile, vectorized, chunked per row to stay under 1 GB
RAM. Then same post‑processing as Tier 0.

### 3.3 Tier 2 — Optional U‑TAE for time prediction

If compute is available, train a small U‑TAE on 64×64 S1 monthly stacks
to predict, per pixel, the **month of change** as a discrete class over
{2020‑01 … today}. Use the argmax month to refine `time_step` for polygons
already produced in Tier 1. This directly targets the Year Accuracy
sub‑metric without risking the main Union IoU.

### 3.4 2020 forest mask

Needed because the challenge defines a post‑2020 deforestation event as
`forest(2020) → non‑forest`. Two options:

1. **AEF‑based** (default, offline) — k‑means on AEF 2020 across all tiles,
   identify the forest cluster by matching high NDVI in 2020 Sentinel‑2.
   Treat the chosen cluster membership as `forest_mask_2020`.
2. **Hansen treecover 2020** (fallback / sanity) — any pixel with ≥ 30 %
   treecover in the 2020 Hansen raster. Used only when AEF is missing.

Predictions outside this mask are zeroed before polygonization.

## 4. Generalization strategy

- AEF is the backbone → trained globally, so the classifier never has to
  learn raw reflectance per region.
- **Leave‑one‑region‑out validation** on training tiles (group by MGRS
  zone or continent) gives a realistic estimate of African performance
  when we only have Amazonian labels.
- All features are **per‑pixel invariants** (differences, temporal
  statistics), not absolute values, so regional biases of S1/S2 are
  partially cancelled.

## 5. What we intentionally skip on day 1

- Training anything that needs the full S3 bucket. The baseline and all
  metrics can be validated on a single downloaded tile or on the
  synthetic mock tile produced by `scripts/generate_mock_data.py`.
- Neural networks with PyTorch/CUDA while on laptop. On the MI300X they are
  the primary submission — see §7.
- Semi‑supervised / pseudo‑labeling loops. Reasonable Tier 3 if leaderboard
  plateaus.

## 6. File map

See [`docs/architecture.md`](./architecture.md) for the module layout.

## 7. Scaling to the AMD MI300X droplet

On the dedicated Ubuntu 24.04 box (1×MI300X, 192 GB HBM, 240 vCPUs, 5 TB
scratch NVMe) we flip the priority: **deep learning becomes the primary
model**, LightGBM drops from "main submission" to "ensemble partner / CPU
utilisation sink". The MI300X's 192 GB HBM + bfloat16 matmul are wasted on
LightGBM, and our 240 vCPUs are wasted on a pure PyTorch pipeline — running
both together extracts value from every part of the machine.

### 7.1 `ChangeUNet` — Tier 1 on the MI300X

An encoder–decoder **U-Net with a transformer bottleneck**, fed the same
per-pixel feature stack as LightGBM (AEF + S1/S2 stats, ~197 channels),
with **two output heads**:

1. `change_head` (sigmoid) — per-pixel change probability. Trained with
   **focal BCE + soft Dice** — Dice directly optimises the Union-IoU the
   leaderboard actually measures.
2. `month_head` (72-way softmax over calendar months from 2020-01) — per-
   pixel month-of-change. Trained with masked cross-entropy on positive
   weak-label pixels only, then reduced to a per-polygon YYMM via expected
   month index. Targets Year Accuracy directly.

Why not replicate U-TAE literally? Our inputs are **annual AEF** + annual
S1/S2 stats — a temporal-attention-over-months design (U-TAE's strength) is
mismatched to annual inputs. A spatial U-Net with an attention bottleneck
is the right shape for our features. If leaderboard progress stalls on
`time_step`, we can slot a true U-TAE behind `month_head` later.

#### 7.1.1 What we actually train on (inputs, targets, weights)

The deep model is trained on **cached patches** extracted from the full
resolution Sentinel‑2 grid (UTM) per tile. We never stream raw GeoTIFF time
series during training because that would under‑utilise the MI300X.

**Inputs (`x`)**

- The same per‑pixel feature vector used by LightGBM, but arranged as a
  dense tensor: `x ∈ R^{F×P×P}` with `F≈197` and patch size `P=256` by
  default.
- Features are built once per tile and stored as `float16` (disk + RAM) and
  converted to `float32` on the fly during training (AMP handles bf16 math).

**Targets**

- `y_change ∈ {0,1}^{P×P}` comes from weak‑label fusion (§3.1), then
  multiplied by the **2020 forest mask** so we only learn the task the
  leaderboard evaluates: `forest(2020) → non‑forest`.
- `y_month ∈ {0..71}^{P×P}` is derived from the fused alert date (median
  across sources per pixel) mapped into a calendar starting at **2020‑01**.
  Month targets are undefined for negative pixels, so we set them to an
  ignore value and do not backprop month loss there.

**Per‑pixel weights**

- `weight ∈ R^{P×P}` is the fused confidence score, used to down‑weight
  ambiguous/low‑confidence positives. This is the key trick for learning
  from noisy labels without collapsing into false positives.

#### 7.1.2 Preprocessing → patch cache (how we keep the GPU busy)

Before training we run a CPU‑heavy preprocessing job that writes a
memory‑mapped cache on the scratch disk:

- `features.npy`  `(F,H,W) float16`
- `labels.npy`    `(H,W) uint8`
- `month.npy`     `(H,W) int16`  (1‑based month index, 0 = no alert)
- `weight.npy`    `(H,W) float16`
- `forest.npy`    `(H,W) uint8`
- `patch_index.jsonl` at the cache root listing valid patches:
  `{tile_id, y, x, positive_fraction}`

This step is embarrassingly parallel across tiles and is where we use the
**240 vCPUs**. Training then becomes “read a small memmapped window, apply
augmentation, feed GPU”.

Patch selection is intentionally biased toward useful signal:

- Skip patches that are almost entirely non‑forest (`forest_min_fraction`).
- Keep a controlled number of hard negatives (all‑forest, all‑negative) so
  the model doesn’t forget the background class.
- Oversample patches with non‑zero `positive_fraction` (rare events).

#### 7.1.3 Losses (and why they match the leaderboard)

We train the two heads jointly with a weighted sum:

\[
L = L_{focal}(change) + \lambda_{dice} L_{dice}(change) + \lambda_{month} L_{CE}(month)
\]

**Change head**

- **Weighted focal BCE** handles extreme class imbalance:
  - Focal focusing reduces the gradient from the many easy negatives.
  - Pixel weights inject weak‑label confidence so noisy positives don’t
    dominate.
- **Soft Dice loss** is included because it directly optimises overlap
  (an IoU‑like objective). Even though the leaderboard measures **Union IoU
  over polygons**, improving contiguous overlap at the pixel level is a
  strong proxy (especially once we apply the shared polygon postprocess).

**Month head**

- **Masked cross‑entropy** is only computed on pixels where `y_change==1`
  (or equivalently where the fused labels say “change happened here”).
- This targets **Year Accuracy** without risking Union IoU: if month is
  wrong, the polygon still exists; only the `time_step` property is off.

Default weights in `configs/server.yaml`:

- `dice_weight = 1.0`
- `month_ce_weight = 0.25`

#### 7.1.4 Optimisation schedule (MI300X‑sized defaults)

The training loop is designed to saturate a single MI300X:

- **Device / AMP**: ROCm bf16 AMP (`torch.autocast` with `bfloat16`).
  (On other GPUs, the runtime falls back to fp16 or fp32 automatically.)
- **Batch size**: autoscaled from VRAM (MI300X default `batch_size=128` for
  `P=256`). This is intentionally conservative; you can raise it if
  utilisation is low.
- **Optimizer**: AdamW (`weight_decay≈1e-4`).
- **LR schedule**: warmup (default 500 steps) + cosine decay to `min_lr`.
- **Gradient clipping**: norm clip 1.0.
- **Epochs**: 40 by default; on new regions we typically resume from
  `best.pt` and fine‑tune for 5–10 epochs.

#### 7.1.5 Validation (for generalisation, not just memorisation)

The goal is to avoid a model that looks great on Amazon tiles but fails in
Africa. We therefore validate with a *region proxy split*:

- Group patches by `tile_id` prefix (first ~5 chars; acts as an MGRS / zone
  proxy).
- Hold out one group (the “last” alphabetically by default) as validation.

This is not perfect, but it is cheap, deterministic, and forces the model
to generalise across spatially distinct tiles. The training loop picks
`best.pt` by validation IoU computed on thresholded change predictions.

#### 7.1.6 Inference details (how we avoid seams + get `time_step`)

Inference is tile‑wise sliding window:

- Patches overlap (`overlap=64` by default) and are stitched with a **Hann
  window overlap‑add**. This removes block seams that otherwise create
  jagged polygons and hurt Union IoU.
- `prob_deep` is the stitched probability raster; we zero it outside the
  2020 forest mask.
- For time, the month head outputs a distribution over months; we compute a
  **soft expected month index** per pixel and later convert to `YYMM`.
  The submission attaches a per‑polygon `time_step` using the existing
  polygon time‑step assignment (median/majority of pixels inside the
  polygon).

#### 7.1.7 How training choices map to leaderboard metrics

- **Union IoU (main)**: Dice + overlap‑blended inference + morphology/area
  filter improve contiguous shapes and reduce boundary noise (polygon
  compactness matters more than pixel speckle).
- **Polygon Recall**: positive‑patch oversampling helps ensure small events
  are seen often enough to be learned.
- **Polygon FPR**: focal loss + confidence weights penalise noisy positives;
  the final shared postprocess enforces minimum area and smooths spurious
  blobs.
- **Year Accuracy**: the month head is trained explicitly for this; errors
  do not remove polygons, so it is “safe” to optimise as a secondary head.

### 7.2 Using the hardware

| Resource                | Role                                                                                          |
|-------------------------|-----------------------------------------------------------------------------------------------|
| **MI300X (192 GB HBM)** | `ChangeUNet` train + inference, bf16 AMP, batch_size 128 (autoscaled from VRAM).              |
| **240 vCPUs**           | Preprocessing (`ProcessPoolExecutor`), LightGBM training + inference (pinned to 236 threads), DataLoader workers. |
| **720 GB boot disk**    | Raw challenge dataset (`data/makeathon-challenge/`).                                           |
| **5 TB scratch NVMe**   | Memory-mapped per-tile caches (features/labels/weights/month/forest), training patch index, checkpoints. |

All defaults come from `src/deforest/runtime.py`, which auto-detects
ROCm/CUDA/CPU and sizes batches/threads/workers accordingly. The same
codebase therefore runs on a MacBook, an NVIDIA dev box, and the MI300X
without any config edits.

### 7.3 Ensemble

Final inference blends `prob_deep` and `prob_gbm` with configurable weights
(`ensemble.weights.{deep,gbm}` in `configs/server.yaml`, default 0.70 / 0.30)
before the shared polygonize / morphology / area-filter / time-step
pipeline. LightGBM tends to preserve small polygons better than the U-Net;
the U-Net cleans up boundaries and captures long-range context. Their
errors are well-correlated with opposite signs and ensembling is a cheap
~1-3 point IoU win typically.

### 7.4 Region generalisation (Africa, SE Asia)

No region-specific priors enter the deep model:

- AEF is a globally trained foundation model and our dominant input.
- S1/S2 features are per-pixel **deltas** (year_last − 2020) that cancel
  regional reflectance biases.
- `ChangeUNet` has ~5.8M params — small enough to not overfit continental
  statistics, large enough to learn polygon morphology cues.
- Validation uses **leave-one-MGRS-prefix-out** by default (see
  `deep/train._split_refs`), which simulates cross-region generalisation
  before we ever touch the test set.
