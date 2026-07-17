# PR 2 static-DFlash DAPO-Math A/B regression

This is a bounded end-to-end systems regression for the SGLang drafter loader
change. It runs one complete trainer epoch over 20 unique DAPO-Math-17k
problems: 2 prompts per step, 4 sampled responses per prompt, and 10 actor
updates with responses capped at 2,048 tokens.

The optimization uses GRPO plus DAPO's Clip-Higher, token-mean loss, and
overlong-reward settings. It does **not** use DAPO dynamic sampling and is not a
DAPO reproduction or a model-quality benchmark.

The paired arms differ in exactly one setting:

- `FMT=null`: preserve SGLang's inherited target loader. With target
  `load_format=dummy`, the external drafter is also initialized with dummy
  weights.
- `FMT=auto`: load the external drafter checkpoint normally while retaining the
  target's hybrid-engine dummy initialization.

## Pinned CUDA 12.8 setup

The experiment branch contains the PR 2 commit plus these helper files. The
formal PR branch remains `agent/fix-sglang-draft-load-format`.

This experiment intentionally reuses the DFlash runtime that already works on
the CUDA 12.8 machine. Do **not** reinstall or reset its SGLang checkout: it
contains the Qwen3.5/GDN/DFlash backports needed by this machine. Also do not
run an unqualified `pip install -e ./verl-v080`; dependency resolution can
replace the tested Transformers, Ray, PyArrow, and CUDA-extension stack.

From the existing outer `verl-SpeCo` checkout, use the known-good interpreter
and source overlays directly:

```bash
export ROOT=/inspire/ssd/project/sais-bio/public/hanchen
export SPECO_ROOT="$ROOT/verl-SpeCo"
export VERL_ROOT="$SPECO_ROOT/verl-v080"
export SGLANG_ROOT="$ROOT/repro-dflash/sglang-dflash22077"
export PYTHON="$ROOT/verl/.venv-verl-dflash/bin/python"
set -o pipefail

# The machine uses a SOCKS proxy. Add only httpx's missing pure-Python extra;
# --no-deps prevents pip from resolving or replacing the pinned runtime stack.
"$PYTHON" -m pip install --no-deps 'socksio==1.0.0'

git -C "$SPECO_ROOT" switch experiment/pr2-static-drafter-ab
git -C "$SPECO_ROOT" pull --ff-only
git -C "$VERL_ROOT" checkout 7aed6b230776f963fa09509c10d9c3a767d1102c

export PYTHONPATH="$SPECO_ROOT:$VERL_ROOT:$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
```

The extra `$SPECO_ROOT/verl-SpeCo` directory created by cloning while already
inside the repository is a duplicate and is not used by these commands.

The enforced runtime versions are Python 3.11, PyTorch 2.9.1+cu128, SGLang
0.5.13, Transformers 5.3.0, Ray 2.55.1, PyArrow 24.0.0, Safetensors 0.8.0,
SocksIO 1.0.0, and TensorDict 0.10.0. Its SGLang base revision is
`f08726fd56c7ff6d8bd258f1545f98148fa4ef58`; the tracked dirty diff SHA-256 is
`5839e45bcf4c6fc85f11385866fe7f1b00bbe973aba941ec5cf7bf4415eb5b71`,
and the required untracked `dflash_timing.py` SHA-256 is
`567cbdf73c6476cfa6aea143bba0f4d2202f7df6d9955cf350771494bbf472c2`.
The manifest SHA-256 over all nine untracked Python files below
`python/sglang` is
`5f2abdd884b7fa935503c27f8e21dc501a54d976edaa337c6b0fbd533705fc61`.
The runner records and enforces these values rather than modifying the
checkout. They do not capture the complete binary environment, so the result
is machine-specific custom-runtime integration evidence, not a reproduction
on stock upstream SGLang. This limitation must be disclosed with the PR
results; an immutable custom-SGLang commit or patch archive is needed before
making a stronger reproducibility claim.

## Prepare a data-independent DAPO-Math subset

The official parquet currently contains repeated rows. The helper streams the
file, deduplicates by prompt text, and selects the 20 smallest SHA-256 ranks of
`speco-pr2-dapo-v1\0<prompt text>`. This makes selection deterministic,
order-independent, and unrelated to either A/B result.

```bash
export EXP_ROOT="$ROOT/pr2-static-drafter-ab"
export DATA_DIR="$EXP_ROOT/data"
mkdir -p "$DATA_DIR"

wget -O "$DATA_DIR/dapo-math-17k.parquet" \
  'https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/65877096c24ffa7abc4e4fa5edb95cf3413a5674/data/dapo-math-17k.parquet?download=true'

echo '534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941  '"$DATA_DIR/dapo-math-17k.parquet" \
  | sha256sum --check

"$PYTHON" "$SPECO_ROOT/experiments/pr2_static_drafter_ab/prepare_dapo20.py" \
  "$DATA_DIR/dapo-math-17k.parquet" \
  "$DATA_DIR/dapo-math-hash20.parquet"
```

The script also writes `dapo-math-hash20.manifest.json`. Keep that manifest with
the logs so the exact prompts and selection rule are auditable.

## Run both arms on 2 x H800

`MODEL_PATH` and `DRAFTER_PATH` may be local directories or Hugging Face model
IDs. Keep every variable and override identical between arms.

For a publishable result, pin the target revision to
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and the DFlash revision to
`9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`, or record the exact revisions of
equivalent local snapshots.

```bash
mkdir -p "$EXP_ROOT/models"
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30

"$PYTHON" -m huggingface_hub.cli.hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir "$EXP_ROOT/models/Qwen3.5-4B" \
  --max-workers 1 &&
  "$PYTHON" -m huggingface_hub.cli.hf download z-lab/Qwen3.5-4B-DFlash \
  --revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
  --local-dir "$EXP_ROOT/models/Qwen3.5-4B-DFlash" \
  --max-workers 1
```

The official endpoint, one download worker, disabled Xet transport, and longer
timeouts avoid the TLS EOF and lock contention seen behind the local proxy.
Both downloads are resumable: rerun this block after a transient failure; do
not delete the partial cache. Do not continue until both commands return zero.

```bash

export MODEL_PATH="$EXP_ROOT/models/Qwen3.5-4B"
export DRAFTER_PATH="$EXP_ROOT/models/Qwen3.5-4B-DFlash"
export TRAIN_FILE="$DATA_DIR/dapo-math-hash20.parquet"
export TEST_FILE="$TRAIN_FILE"
export OUT="$EXP_ROOT/results"
mkdir -p "$OUT"

"$PYTHON" "$SPECO_ROOT/experiments/pr2_static_drafter_ab/preflight.py" \
  | tee "$OUT/preflight.manual.json" &&
  { pkill gg || true; } &&
  nvidia-smi &&
  FMT=null bash "$SPECO_ROOT/experiments/pr2_static_drafter_ab/run_one.sh" &&
  FMT=auto bash "$SPECO_ROOT/experiments/pr2_static_drafter_ab/run_one.sh"
```

The command chain releases the `gg` reservation only after preflight succeeds.
Preflight verifies both exact Hub revisions, every expected weight shard, and
the absence of incomplete download files before touching the GPUs. Training
then enables Hugging Face and Transformers offline modes so a proxy failure
cannot perturb either arm.

Each arm runs the same lightweight, import-free preflight again and stores its
report as `<run>/preflight.json`. Stop if its status is `error`, if a module
resolves outside the three configured source roots, or if a reference SGLang
hash is unexpected. Hash drift is fatal by default;
`ALLOW_SGLANG_DRIFT=1` is only for an intentional, separately documented
runtime and should never be introduced
between arms. The preflight also requires clean tracked SpeCo and VeRL trees
and records the exact experiment commit. Absence of `flash_attn.bert_padding`
is recorded but is not fatal,
because this configuration disables remove-padding and forces SDPA. A failure
in that import during the first actor update would identify an additional VeRL
compatibility path instead of a drafter-loader failure.

On a dedicated machine, the script clears stale local Ray state before each
arm. Set `RESET_RAY=0` if that machine is intentionally attached to a shared
Ray cluster. A reasonable 2 x H800 estimate is 30--65 minutes per arm and
1.25--1.8 hours for the pair; the dummy-drafter arm is expected to be slower.

## Summarize

```bash
"$PYTHON" "$SPECO_ROOT/experiments/pr2_static_drafter_ab/summarize.py" \
  "$OUT/qwen35-dflash-dapo-math20-null/train.log" \
  "$OUT/qwen35-dflash-dapo-math20-auto/train.log"
```

A valid systems regression must complete all 10 updates and show a nonzero
gradient or a nonzero GRPO advantage range. If every group is uniformly wrong
or uniformly correct, report the run as inconclusive instead of changing the
subset after seeing the A/B results.

For the PR, report loader evidence, update count, gradient/advantage evidence,
acceptance length/rate, and warm generation-time medians. Reward is only a
sanity check because stochastic samples need not match across arms. Do not
describe the timing difference as DFlash versus autoregressive decoding: this
A/B specifically tests real-checkpoint versus dummy-checkpoint drafter loading.
