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

## Clean setup

The experiment branch contains the PR 2 commit plus these helper files. The
formal PR branch remains `agent/fix-sglang-draft-load-format`.

```bash
git clone --branch experiment/pr2-static-drafter-ab \
  https://github.com/1ring2rta/verl-SpeCo.git

git clone https://github.com/verl-project/verl.git verl-v080
git -C verl-v080 checkout 7aed6b230776f963fa09509c10d9c3a767d1102c
python -m pip install -e ./verl-v080
```

Use a clean SGLang >= 0.5.12 environment. The upstream SpeCo Dockerfile uses
`verlai/verl:sgl0512.dev1`; using that image and then installing the pinned VeRL
checkout above gives the cleanest PR evidence.

## Prepare a data-independent DAPO-Math subset

The official parquet currently contains repeated rows. The helper streams the
file, deduplicates by prompt text, and selects the 20 smallest SHA-256 ranks of
`speco-pr2-dapo-v1\0<prompt text>`. This makes selection deterministic,
order-independent, and unrelated to either A/B result.

```bash
export SPECO_ROOT=$PWD/verl-SpeCo
export VERL_ROOT=$PWD/verl-v080
export DATA_DIR=$PWD/data
mkdir -p "$DATA_DIR"

wget -O "$DATA_DIR/dapo-math-17k.parquet" \
  'https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/65877096c24ffa7abc4e4fa5edb95cf3413a5674/data/dapo-math-17k.parquet?download=true'

echo '534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941  '"$DATA_DIR/dapo-math-17k.parquet" \
  | sha256sum --check

python "$SPECO_ROOT/experiments/pr2_static_drafter_ab/prepare_dapo20.py" \
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
mkdir -p "$PWD/models"
hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --local-dir "$PWD/models/Qwen3.5-4B"
hf download z-lab/Qwen3.5-4B-DFlash \
  --revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf \
  --local-dir "$PWD/models/Qwen3.5-4B-DFlash"

export MODEL_PATH=$PWD/models/Qwen3.5-4B
export DRAFTER_PATH=$PWD/models/Qwen3.5-4B-DFlash
export TRAIN_FILE=$DATA_DIR/dapo-math-hash20.parquet
export OUT=$PWD/pr2-ab-results

FMT=null bash "$SPECO_ROOT/experiments/pr2_static_drafter_ab/run_one.sh"
FMT=auto bash "$SPECO_ROOT/experiments/pr2_static_drafter_ab/run_one.sh"
```

On a dedicated machine, the script clears stale local Ray state before each
arm. Set `RESET_RAY=0` if that machine is intentionally attached to a shared
Ray cluster. A reasonable 2 x H800 estimate is 30--65 minutes per arm and
1.25--1.8 hours for the pair; the dummy-drafter arm is expected to be slower.

## Summarize

```bash
python "$SPECO_ROOT/experiments/pr2_static_drafter_ab/summarize.py" \
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
