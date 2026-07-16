#!/usr/bin/env bash
set -euo pipefail

: "${SPECO_ROOT:?set SPECO_ROOT to the verl-SpeCo checkout}"
: "${VERL_ROOT:?set VERL_ROOT to VeRL commit 7aed6b230776f963fa09509c10d9c3a767d1102c}"
: "${SGLANG_ROOT:?set SGLANG_ROOT to the tested custom SGLang checkout}"
: "${PYTHON:?set PYTHON to the tested DFlash Python interpreter}"
: "${MODEL_PATH:?set MODEL_PATH to Qwen3.5-4B}"
: "${DRAFTER_PATH:?set DRAFTER_PATH to Qwen3.5-4B-DFlash}"
: "${TRAIN_FILE:?set TRAIN_FILE to the fixed 20-row train parquet}"
: "${OUT:?set OUT to the result directory}"
: "${FMT:?set FMT to null or auto}"

if [[ "$FMT" != "null" && "$FMT" != "auto" ]]; then
  echo "FMT must be null or auto, got: $FMT" >&2
  exit 2
fi

RESET_RAY=${RESET_RAY:-1}
TEST_FILE=${TEST_FILE:-$TRAIN_FILE}
RUN="qwen35-dflash-dapo-math20-${FMT}"
RUN_DIR="$OUT/$RUN"
mkdir -p "$RUN_DIR"

export SPECO_ROOT VERL_ROOT SGLANG_ROOT PYTHON MODEL_PATH DRAFTER_PATH TRAIN_FILE TEST_FILE
export PYTHONPATH="$SPECO_ROOT:$VERL_ROOT:$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export VERL_SPECO_STRICT_VERL=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export USE_HUB_KERNELS=NO
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

"$PYTHON" "$SPECO_ROOT/experiments/pr2_static_drafter_ab/preflight.py" \
  | tee "$RUN_DIR/preflight.json"

if [[ "$RESET_RAY" == "1" ]]; then
  "$PYTHON" -m ray.scripts.scripts stop --force || true
fi

"$PYTHON" -m verl_speco.main \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=2 \
  data.train_max_samples=-1 \
  data.val_max_samples=2 \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=1 \
  data.truncation=error \
  data.shuffle=False \
  data.seed=42 \
  data.dataloader_num_workers=1 \
  +data.apply_chat_template_kwargs.enable_thinking=true \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.mtp.enable=True \
  actor_rollout_ref.model.mtp.enable_train=False \
  actor_rollout_ref.model.mtp.enable_rollout=True \
  actor_rollout_ref.model.mtp.speculative_algorithm=DFLASH \
  actor_rollout_ref.model.mtp.speculative_num_steps=1 \
  actor_rollout_ref.model.mtp.speculative_eagle_topk=1 \
  actor_rollout_ref.model.mtp.speculative_num_draft_tokens=16 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.offload_policy=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.max_num_seqs=2 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
  actor_rollout_ref.rollout.drafter.enable=True \
  actor_rollout_ref.rollout.drafter.enable_drafter_training=False \
  actor_rollout_ref.rollout.drafter.model_path="$DRAFTER_PATH" \
  actor_rollout_ref.rollout.drafter.speculative_algorithm=DFLASH \
  actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
  actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
  actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=16 \
  actor_rollout_ref.rollout.drafter.rollout.cuda_graph_max_bs=2 \
  actor_rollout_ref.rollout.drafter.sglang.draft_load_format="$FMT" \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.log_level=info \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=42 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=1 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=8192 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.max_mamba_cache_size=8 \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_custom_all_reduce=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.enforce_disable_flashinfer_allreduce_fusion=True \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=extra_buffer \
  reward.reward_manager.name=dapo \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=true \
  +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward.reward_kwargs.overlong_buffer_cfg.log=true \
  +reward.reward_kwargs.max_resp_len=2048 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.logger='["console"]' \
  trainer.project_name=verl-speco-pr2 \
  trainer.experiment_name="$RUN" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  trainer.balance_batch=False \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$RUN_DIR/ckpt" \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  "$@" \
  2>&1 | tee "$RUN_DIR/train.log"
