# PR 2 static-DFlash A/B regression

This is a bounded end-to-end systems regression for the SGLang drafter loader
change. It runs one complete trainer epoch over a fixed 20-row GSM8K subset:
2 prompts per step, 4 sampled responses per prompt, and 10 actor updates.

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

## Prepare the controlled dataset

```bash
export SPECO_ROOT=$PWD/verl-SpeCo
export VERL_ROOT=$PWD/verl-v080
export DATA_DIR=$PWD/data

python "$VERL_ROOT/examples/data_preprocess/gsm8k.py" \
  --local_save_dir "$DATA_DIR/gsm8k"

python "$SPECO_ROOT/experiments/pr2_static_drafter_ab/prepare_hard20.py" \
  "$DATA_DIR/gsm8k/train.parquet" \
  "$DATA_DIR/gsm8k-hard20/train.parquet"
```

The helper selects four known nontrivial source rows and repeats them five
times. Repetition is intentional: this is a controlled systems test designed
to produce real policy gradients across all 10 updates, not a quality
evaluation.

## Run both arms on 2 x H800

`MODEL_PATH` and `DRAFTER_PATH` may be local directories or Hugging Face model
IDs. Keep every variable and override identical between arms.

```bash
export MODEL_PATH=Qwen/Qwen3.5-4B
export DRAFTER_PATH=z-lab/Qwen3.5-4B-DFlash
export TRAIN_FILE=$DATA_DIR/gsm8k-hard20/train.parquet
export TEST_FILE=$DATA_DIR/gsm8k/test.parquet
export OUT=$PWD/pr2-ab-results

FMT=null bash "$SPECO_ROOT/experiments/pr2_static_drafter_ab/run_one.sh"
FMT=auto bash "$SPECO_ROOT/experiments/pr2_static_drafter_ab/run_one.sh"
```

On a dedicated machine, the script clears stale local Ray state before each
arm. Set `RESET_RAY=0` if that machine is intentionally attached to a shared
Ray cluster. A local run of the same scale took roughly 18 minutes per arm on
2 x H800; allow 36--45 minutes for the pair including initialization.

## Summarize

```bash
python "$SPECO_ROOT/experiments/pr2_static_drafter_ab/summarize.py" \
  "$OUT/qwen35-dflash-dapo-style-null/train.log" \
  "$OUT/qwen35-dflash-dapo-style-auto/train.log"
```

For the PR, report loader evidence, completed update count, nonzero gradient,
acceptance length/rate, and warm generation-time medians. Reward is only a
sanity check because stochastic samples need not match across arms. Do not
describe the timing difference as DFlash versus autoregressive decoding: this
A/B specifically tests real-checkpoint versus dummy-checkpoint drafter loading.
