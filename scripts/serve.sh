#!/bin/bash
# Host wrapper: launch the serving container on ONE DGX Spark. Detached.
#
#   bash scripts/serve.sh              # measured configuration
#   DRY_RUN=1 bash scripts/serve.sh    # print the docker command, run nothing
#
# Requires passwordless docker (sudo -n), the pack downloaded and verified, and
# the alias side file built (scripts/build_cold_alias.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

DRY_RUN="${DRY_RUN:-0}"
TS=$(date +%Y%m%d-%H%M%S)
RUN="${RUN_DIR:-$RUNS_DIR/$TS}"
JIT="$CACHE_DIR/kkref/jit"

# The container gets $RUN/ctr only: guard.sh runs as root and reads/appends files in $RUN,
# so the container must not be able to swap those for symlinks.
mkdir -p "$RUN/ctr" "$JIT"/{b12x/compile,b12x/cute,cute-dsl,tilelang,triton,nv-compute,vllm-r5o}

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: would launch container=$CONTAINER image=$IMAGE"
  echo "  model  : $MODEL_DIR -> /models/$MODEL_NAME"
  echo "  alias  : $ALIAS_DIR -> /alias"
  echo "  run dir: $RUN/ctr -> /mem"
  echo "  port   : $API_PORT"
  echo "  maxlen : $MAXLEN  kv=$KVB  mnbt=$MNBT  lpt=$LPT  mtp_k=$MTP_K"
  echo "  graphs : $CG_MODE sizes=$CAPTURE_SIZES  gpu_util=$GPU_MEM_UTIL"
  echo "  ats    : layers=$ATS_LAYERS mtp=$ATS_MTP embed=$ATS_EMBED"
  echo
  echo "env:"
  echo "  B12X_AUTOTUNE=0"
  echo "  B12X_BIND_COMPILE_WORKERS=2 B12X_STATE_COMPILE_WORKERS=2 B12X_WEIGHTS_COMPILE_WORKERS=4"
  echo "  B12X_DENSE_SPLITK_TURBO=1 B12X_W4A8_TINY_DECODE=0"
  echo "  VLLM_DS41_MARKOV_NVFP4=1 VLLM_DS41_L2_PREFETCH=0 VLLM_DS41_ROPE_TRIM=1"
  echo "  VLLM_EXL3_MIXEDK_MODE=unified VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_V41_NATIVE_MOE=0"
  echo "  VLLM_EXL3_ATS_ALIAS=1 VLLM_EXL3_ATS_ALIAS_DIR=/alias"
  echo "  VLLM_EXL3_ATS_ALIAS_LAYERS=$ATS_LAYERS VLLM_EXL3_ATS_ALIAS_MTP=$ATS_MTP VLLM_EXL3_ATS_ALIAS_EMBED=$ATS_EMBED"
  echo "  VLLM_DS41_ATTENTION_COMPUTE=auto VLLM_DS41_KERNEL_BACKEND=tilelang VLLM_DS41_HEAD_DTYPE=float32"
  echo "  VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  echo
  echo "vllm serve /models/$MODEL_NAME --served-model-name dsv41 --tokenizer-mode deepseek_v41 \\"
  echo "  --dtype bfloat16 --tensor-parallel-size 1 --distributed-executor-backend uni \\"
  echo "  --quantization exl3_dsv41 --block-size 256 --no-enable-prefix-caching --enable-chunked-prefill \\"
  echo "  --max-model-len $MAXLEN --max-num-seqs 1 --max-parallel-prefills 1 \\"
  echo "  --max-num-batched-tokens $MNBT --gpu-memory-utilization $GPU_MEM_UTIL \\"
  echo "  --kv-cache-memory-bytes $KVB --generation-config vllm --attention-backend tilelang \\"
  echo "  --linear-backend tilelang --moe-backend tilelang --jit-monitor-mode warn \\"
  echo "  --compilation-config {\"cudagraph_mode\":\"$CG_MODE\",...,\"cudagraph_capture_sizes\":[$CAPTURE_SIZES]} \\"
  echo "  --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$MTP_K,...} \\"
  echo "  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \\"
  echo "  --long-prefill-token-threshold $LPT --async-scheduling --no-scheduler-reserve-full-isl \\"
  echo "  --no-enable-flashinfer-autotune --language-model-only --mm-processor-cache-gb 0 \\"
  echo "  --hf-overrides {...exl3 routed experts + deepseek_v41_fp8 non-routed...}"
  exit 0
fi

[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR" >&2; exit 1; }
[ -f "$ALIAS_DIR/experts.bin" ] || { echo "alias side file not found: $ALIAS_DIR/experts.bin (run build_cold_alias.sh)" >&2; exit 1; }
[ -f "$ALIAS_DIR/manifest.json" ] || { echo "alias manifest not found: $ALIAS_DIR/manifest.json" >&2; exit 1; }
sudo -n docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not found: $IMAGE (run build_image.sh)" >&2; exit 1; }

# ---- host memory policy (restored by guard.sh stop) ------------------------
SYSCTL_KEYS="vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness"
: >"$RUN/sysctl.before"
for k in $SYSCTL_KEYS; do
  printf '%s=%s\n' "$k" "$(cat "/proc/sys/${k//.//}")" >>"$RUN/sysctl.before"
done

sudo -n sysctl -q vm.compaction_proactiveness=0 vm.watermark_boost_factor=0 vm.swappiness=0
sudo -n swapoff -a && sudo -n swapon -a
sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'
sleep 3

BASE_SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo)
echo "MemAvailable: $(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB  swap_base: $BASE_SWAP KiB"

sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

sudo -n docker run -d --name "$CONTAINER" \
  --gpus all \
  --memory=118g --memory-swap=118g --oom-score-adj 1000 --ulimit core=0 \
  --shm-size=2g \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --ulimit=memlock=-1:-1 \
  -v "$MODELS_DIR":/models:ro \
  -v "$CACHE_DIR":/cache:rw \
  -v "$ALIAS_DIR":/alias:ro \
  -v "$RUN/ctr":/mem:rw \
  -v "$HERE/serve_inner.sh":/root/serve.sh:ro \
  -p "$API_PORT":8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e B12X_AUTOTUNE=0 -e B12X_BIND_COMPILE_WORKERS=2 -e B12X_STATE_COMPILE_WORKERS=2 \
  -e B12X_WEIGHTS_COMPILE_WORKERS=4 \
  -e B12X_COMPILE_CACHE_DIR="$JIT/b12x/compile" \
  -e B12X_CUTE_COMPILE_CACHE_DIR="$JIT/b12x/cute" \
  -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A8_TINY_DECODE=0 \
  -e CUDA_CACHE_MAXSIZE=4294967296 -e CUDA_CACHE_PATH="$JIT/nv-compute" \
  -e CUTE_DSL_ARCH=sm_121a -e CUTE_DSL_CACHE_DIR="$JIT/cute-dsl" \
  -e TILELANG_CACHE_DIR="$JIT/tilelang" -e TRITON_CACHE_DIR="$JIT/triton" \
  -e XDG_CACHE_HOME="$JIT" -e VLLM_CACHE_ROOT="$JIT/vllm-r5o" \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_DS41_ATTENTION_COMPUTE=auto \
  -e VLLM_DS41_KERNEL_BACKEND=tilelang \
  -e VLLM_DS41_HEAD_DTYPE=float32 \
  -e VLLM_DS41_PACKED_BF16_LM_HEAD=0 \
  -e VLLM_L2_PREFETCH_KERNELS=tilelang \
  -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_DISABLE_PYNCCL=0 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
  -e VLLM_EXL3_MOE_KERNEL=exllamav3 \
  -e VLLM_EXL3_V41_NATIVE_MOE=0 \
  -e VLLM_EXL3_MIXEDK_LEGACY=1 \
  -e VLLM_EXL3_MIXEDK_MODE=unified \
  -e VLLM_EXL3_TRIM_AFTER_LAYER=1 \
  -e VLLM_EXL3_MODEL_DIR="/models/$MODEL_NAME" \
  -e VLLM_EXL3_ATS_ALIAS=1 \
  -e VLLM_EXL3_ATS_ALIAS_DIR=/alias \
  -e VLLM_EXL3_ATS_ALIAS_LAYERS="$ATS_LAYERS" \
  -e VLLM_EXL3_ATS_ALIAS_MTP="$ATS_MTP" \
  -e VLLM_EXL3_ATS_ALIAS_EMBED="$ATS_EMBED" \
  -e VLLM_EXL3_ATS_PREFETCH=1 \
  -e VLLM_DS41_ROPE_TRIM=1 \
  -e VLLM_DS41_MARKOV_NVFP4=1 \
  -e VLLM_DS41_L2_PREFETCH=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e DSV41_MEM_PROBE=1 \
  -e VLLM_EXL3_MEM_WATERFALL=0 \
  -e VLLM_EXL3_MEM_WATERFALL_PATH= \
  -e MAXLEN="$MAXLEN" -e KVB="$KVB" -e MNBT="$MNBT" -e LPT="$LPT" \
  -e MTP_K="$MTP_K" -e CAPTURE_SIZES="$CAPTURE_SIZES" -e CG_MODE="$CG_MODE" \
  -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  --entrypoint /bin/bash \
  "$IMAGE" /root/serve.sh \
  >"$RUN/container_id.txt" 2>&1

echo "LAUNCHED $CONTAINER ($(cat "$RUN/container_id.txt"))"
echo "run dir: $RUN"

# ---- memory guard ---------------------------------------------------------
echo "preflight" >"$RUN/stage"
sudo -n setsid nohup bash "$HERE/guard.sh" watch "$CONTAINER" "$RUN/stage" "$RUN/guard.log" 524288 "$BASE_SWAP" \
  >/dev/null 2>&1 &
echo "guard armed: MemAvailable floor 0.5 GiB, swap growth <= 64 MiB, 0.05 s"
echo "watch: bash scripts/guard.sh status   stop: bash scripts/guard.sh stop"