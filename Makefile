SHELL := /bin/bash
PY    := .venv/bin/python

# ----------------------------------------------------------------------------
# Server knobs. Override on the command line, e.g.
#     make train-deep CONFIG=configs/server.yaml CACHE=/mnt/scratch/dfp
# ----------------------------------------------------------------------------
CONFIG       ?= configs/default.yaml
SERVER_CFG   ?= configs/server.yaml
CACHE        ?= /mnt/scratch/deforest/patches
CHECKPOINTS  ?= /mnt/scratch/deforest/checkpoints
SUBMISSION   ?= submissions/submission.geojson
DEEP_CKPT    ?= $(CHECKPOINTS)/best.pt
GBM_MODEL    ?= models/gbm.txt
# ROCm wheel index for the MI300X droplet. Swap for the CUDA index on an NVIDIA box.
TORCH_INDEX  ?= https://download.pytorch.org/whl/rocm6.2
TORCH_PKGS   ?= torch==2.5.1 torchvision==0.20.1

.PHONY: help install install-gpu mock baseline train-gbm submit evaluate \
        preprocess train-deep submit-ensemble runtime test clean

help:
	@echo "Local / laptop targets:"
	@echo "  install            Create .venv and install CPU requirements"
	@echo "  mock               Generate a synthetic tile under data/makeathon-challenge/"
	@echo "  baseline           Produce a submission with the zero-training consensus model"
	@echo "  train-gbm          Train the LightGBM Tier-1 model"
	@echo "  submit             Produce a LightGBM submission (CPU-only)"
	@echo "  evaluate           Local metrics against training labels"
	@echo "  test               Run unit tests"
	@echo ""
	@echo "Server / MI300X targets (CONFIG=configs/server.yaml):"
	@echo "  install-gpu        Install ROCm PyTorch + GPU extras on top of install"
	@echo "  runtime            Print detected hardware + autoscaled defaults"
	@echo "  preprocess         Pre-compute per-tile feature caches on /mnt/scratch"
	@echo "  train-deep         Train the ChangeUNet deep model"
	@echo "  submit-ensemble    Ensemble deep + GBM → final submission.geojson"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.venv:
	python3 -m venv .venv
	.venv/bin/pip install -U pip uv

install: .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

install-gpu: install
	.venv/bin/pip install --index-url $(TORCH_INDEX) $(TORCH_PKGS)
	.venv/bin/pip install -r requirements-gpu.txt

# ---------------------------------------------------------------------------
# Laptop pipeline (unchanged)
# ---------------------------------------------------------------------------

mock: install
	$(PY) scripts/generate_mock_data.py --out data/makeathon-challenge --tile MOCK_0_0

baseline: install
	$(PY) scripts/build_submission.py --model baseline --config $(CONFIG) \
	    --out submissions/baseline.geojson

train-gbm: install
	$(PY) scripts/train_gbm.py --config $(CONFIG) --out $(GBM_MODEL)

submit: install
	$(PY) scripts/build_submission.py --model gbm --config $(CONFIG) \
	    --gbm-model $(GBM_MODEL) --out submissions/gbm.geojson

evaluate: install
	$(PY) scripts/evaluate.py --config $(CONFIG) \
	    --predictions submissions/baseline.geojson

# ---------------------------------------------------------------------------
# Server pipeline (MI300X)
# ---------------------------------------------------------------------------

runtime: install
	$(PY) -m deforest.cli runtime

preprocess: install
	$(PY) scripts/preprocess_tiles.py --config $(SERVER_CFG) --split train --cache-dir $(CACHE)
	$(PY) scripts/preprocess_tiles.py --config $(SERVER_CFG) --split test  --cache-dir $(CACHE)

train-deep: install
	$(PY) scripts/train_deep.py --config $(SERVER_CFG) \
	    --cache-dir $(CACHE) --checkpoint-dir $(CHECKPOINTS)

submit-ensemble: install
	$(PY) scripts/predict_ensemble.py --config $(SERVER_CFG) \
	    --deep-ckpt $(DEEP_CKPT) --gbm-model $(GBM_MODEL) \
	    --split test --cache-dir $(CACHE) --out $(SUBMISSION)

# ---------------------------------------------------------------------------

test: install
	.venv/bin/pytest

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
