# Running the deforestation pipeline on Ubuntu 24.04 + AMD MI300X

Target droplet

| Resource       | Value                  |
|----------------|------------------------|
| GPU            | 1 × AMD Instinct MI300X |
| VRAM           | 192 GB HBM3             |
| CPU            | ~240 vCPUs              |
| System RAM     | ~1.8 TB                 |
| Boot disk      | 720 GB NVMe             |
| Scratch disk   | 5 TB NVMe (mount at `/mnt/scratch`) |

The laptop/CPU pipeline stays unchanged; this guide layers a GPU-saturating
deep-learning stack on top. Both models run in parallel and are ensembled
at submission time — the deep model carries most of the Union-IoU weight,
the LightGBM model contributes pixel-precise residuals.

---

## 1. OS setup

```bash
# ROCm 6.x (required for PyTorch on AMD). Follow AMD's official guide:
#   https://rocm.docs.amd.com/projects/install-on-linux/
#
# Quick summary for Ubuntu 24.04:
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential git wget
sudo wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/noble/amdgpu-install_6.2.60200-1_all.deb
sudo apt install -y ./amdgpu-install_6.2.60200-1_all.deb
sudo amdgpu-install --usecase=rocm --no-dkms
sudo usermod -aG video,render $USER
newgrp render                                   # take effect without logging out
rocm-smi                                        # sanity check — should list MI300X

# LightGBM on Linux just needs libgomp, already present with build-essential.
# System libs for rasterio / geopandas:
sudo apt install -y libgdal-dev gdal-bin libproj-dev proj-data proj-bin \
                    libspatialindex-dev libgeos-dev
```

## 2. Attach the scratch disk

```bash
lsblk                                           # find the 5 TB NVMe, e.g. /dev/nvme1n1
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /mnt/scratch
sudo mount /dev/nvme1n1 /mnt/scratch
sudo chown -R $USER:$USER /mnt/scratch
echo "/dev/nvme1n1 /mnt/scratch ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

The pipeline reads `DEFOREST_SCRATCH` (defaulting to `/mnt/scratch`) and
auto-creates `/mnt/scratch/deforest/{patches,checkpoints}` on first run.

## 3. Project install

```bash
git clone <this-repo>
cd make
make install-gpu TORCH_INDEX=https://download.pytorch.org/whl/rocm6.2
```

Sanity check:

```bash
make runtime
# device=rocm (AMD Instinct MI300X) × 1, vram=192 GiB/gpu, cpu=240 cores,
# ram=… scratch=/mnt/scratch (4900 GiB free), bf16=True
# autoscaled defaults: batch_size=128 amp_dtype=bfloat16 num_workers=30 …
```

## 4. Dataset

Download the challenge dataset into `data/makeathon-challenge/` using the
`download_data.py` shipped in the challenge repo. The 720 GB boot disk fits
the full dataset; the 5 TB scratch holds the preprocessed tensors.

## 5. End-to-end run

```bash
# 1. Pre-compute per-tile caches (parallelised across all 240 vCPUs).
make preprocess
# → /mnt/scratch/deforest/patches/<tile>/ + patch_index.jsonl

# 2. Train the deep ChangeUNet (bf16 AMP on the MI300X).
make train-deep
# → /mnt/scratch/deforest/checkpoints/best.pt

# 3. Train the LightGBM CPU model on the same features (full 240 vCPUs).
make train-gbm CONFIG=configs/server.yaml

# 4. Ensemble both models over the test split → final GeoJSON.
make submit-ensemble
# → submissions/submission.geojson
```

## 6. How the server is actually used

| Resource                  | Where                                        |
|---------------------------|----------------------------------------------|
| **MI300X (192 GB HBM)**   | `ChangeUNet` training + inference, bf16 AMP, batch_size=128 |
| **240 vCPUs**             | Preprocessing (`ProcessPoolExecutor`), LightGBM training & inference, DataLoader workers |
| **720 GB boot NVMe**      | Raw Sentinel dataset, virtualenv, git checkout |
| **5 TB scratch NVMe**     | Memory-mapped feature caches, model checkpoints |

Autoscaling is centralised in `src/deforest/runtime.py`. The only config
values you might want to touch per run:

* `deep.batch_size`, `deep.patch_size` — trade VRAM for throughput.
* `deep.epochs`, `deep.lr` — training schedule length.
* `ensemble.weights.deep` / `.gbm` — shift the blend towards whichever
  model validates better on your held-out region.

## 7. Generalisation to other regions (Africa, SE Asia)

The training recipe is region-agnostic because:

* **Features are region-invariant** — AEF embeddings are globally trained,
  S1/S2 stats are per-pixel normalised via annual deltas.
* **Weak labels cover every training region** — RADD is global, GLAD-L
  covers tropics, GLAD-S2 is pantropical. The fusion step auto-adapts to
  whichever subset is available for a given tile.
* **The 2020 forest mask** comes from AEF + NDVI and works wherever AEF
  exists (worldwide).
* **`ChangeUNet` is a plain U-Net with attention bottleneck** — no region-
  specific priors are baked in.

For a fresh region you only need to:

1. Download imagery + labels into `data/makeathon-challenge/...<region>...`
2. Re-run `make preprocess train-deep train-gbm submit-ensemble`.
3. Optionally, fine-tune with `train-deep --resume checkpoints/best.pt`.

## 8. Troubleshooting

* `torch.cuda.is_available() == False` on MI300X:
  `rocminfo | head`, check `HIP_VISIBLE_DEVICES`, verify the ROCm wheel is
  actually installed (`python -c "import torch; print(torch.version.hip)"`).
* Out-of-memory during training: lower `deep.batch_size` in
  `configs/server.yaml`. 128 is the autoscaled default; the model happily
  fits at 64.
* LightGBM "num_threads" warning: safe to ignore — we explicitly pin the
  count via the wrapper.
* Slow data loading: increase `preprocess.workers` to saturate CPU
  bandwidth while preprocessing, and `deep.num_workers` during training.
