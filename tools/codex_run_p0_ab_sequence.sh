#!/usr/bin/env bash
# AI-generated, awaiting verification by recoletas on 2026-07-09.
#
# Run inside the SCNet container. Executes a short P0 A/B sequence across the
# current safe candidate branches. This script deliberately restarts vLLM for
# each case because the tested env knobs are read by the server process at
# import/start time.

set -euo pipefail

Q="${Q:-/public/home/xdzs2026_c087}"
GOVINDA_DIR="${GOVINDA_DIR:-$Q/Govinda}"
RUN_PROFILE="${RUN_PROFILE:-quick}"
OUT_ROOT="${OUT_ROOT:-$Q/testdata/test/codex_p0_ab_$(date +%Y%m%d_%H%M%S)}"
WAIT_HEALTH_TIMEOUT_S="${WAIT_HEALTH_TIMEOUT_S:-3600}"
REMOTE_URL="${REMOTE_URL:-}"

echo "=== P0 A/B sequence $(date +%F_%T) ==="
echo "RUN_PROFILE=$RUN_PROFILE"
echo "OUT_ROOT=$OUT_ROOT"
echo "REMOTE_URL=${REMOTE_URL:+set}"

case_lines_quick=(
  "mid_4_8|experiment/tile64-mid-default-20260709|mid|16|64|0|ieee|0|4-8K|3"
  "mid_8_16|experiment/tile64-mid-default-20260709|mid|16|64|0|ieee|0|8-16K|3"
  "mid_gdn16_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|0|ieee|0|4-8K|3"
  "mid_gdn16_fla_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|1|ieee|0|4-8K|3"
)

case_lines_gdn=(
  "mid_gdn8_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|8|64|0|ieee|0|4-8K|3"
  "mid_gdn16_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|0|ieee|0|4-8K|3"
  "mid_gdn32_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|32|64|0|ieee|0|4-8K|3"
  "mid_gdn16_chunk32_4_8|experiment/tile64-mid-gdn-chunk-20260709|mid|16|32|0|ieee|0|4-8K|3"
  "mid_gdn16_tf32_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|0|tf32|0|4-8K|3"
  "mid_gdn16_fla_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|1|ieee|0|4-8K|3"
)

case_lines_decode=(
  "mid_decode_base_4_8|experiment/tile64-mid-default-20260709|mid|16|64|0|ieee|0|4-8K|3"
  "mid_llmm1silu_4_8|experiment/mid-llmm1silu-20260709|mid|16|64|0|ieee|1|4-8K|3"
  "mid_llmm1silu_8_16|experiment/mid-llmm1silu-20260709|mid|16|64|0|ieee|1|8-16K|3"
)

case_lines_full=(
  "mid_4_8|experiment/tile64-mid-default-20260709|mid|16|64|0|ieee|0|4-8K|3"
  "mid_8_16|experiment/tile64-mid-default-20260709|mid|16|64|0|ieee|0|8-16K|3"
  "mid_16_32|experiment/tile64-mid-default-20260709|mid|16|64|0|ieee|0|16-32K|3"
  "mid_gdn8_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|8|64|0|ieee|0|4-8K|3"
  "mid_gdn16_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|0|ieee|0|4-8K|3"
  "mid_gdn16_chunk32_4_8|experiment/tile64-mid-gdn-chunk-20260709|mid|16|32|0|ieee|0|4-8K|3"
  "mid_gdn16_tf32_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|0|tf32|0|4-8K|3"
  "mid_gdn16_fla_4_8|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|1|ieee|0|4-8K|3"
  "mid_gdn16_8_16|experiment/tile64-mid-gdn-conv-20260709|mid|16|64|0|ieee|0|8-16K|3"
  "mid_llmm1silu_4_8|experiment/mid-llmm1silu-20260709|mid|16|64|0|ieee|1|4-8K|3"
)

case "$RUN_PROFILE" in
  quick)
    selected_cases=("${case_lines_quick[@]}")
    ;;
  gdn)
    selected_cases=("${case_lines_gdn[@]}")
    ;;
  decode)
    selected_cases=("${case_lines_decode[@]}")
    ;;
  full)
    selected_cases=("${case_lines_full[@]}")
    ;;
  *)
    echo "ERROR: RUN_PROFILE must be quick, gdn, decode, or full" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_ROOT"
summary="$OUT_ROOT/summary.tsv"
printf "case\tref\tpolicy\tgdn_block\tgdn_chunk\tfla_fix_bt\ttril_precision\tllmm1silu\trange\tnum_prompts\tstatus\tresult_dir\n" \
  > "$summary"

for line in "${selected_cases[@]}"; do
    IFS='|' read -r label ref tile_policy gdn_block gdn_chunk fla_fix tril_precision llmm1silu range num_prompts <<< "$line"
    case_out="$OUT_ROOT/$label"
    mkdir -p "$case_out"

    echo
    echo "=== case $label ==="
    echo "ref=$ref policy=$tile_policy gdn_block=$gdn_block gdn_chunk=$gdn_chunk fla_fix=$fla_fix tril_precision=$tril_precision llmm1silu=$llmm1silu range=$range prompts=$num_prompts"

    set +e
    REMOTE_URL="$REMOTE_URL" \
    REF="$ref" \
    VLLM_TRITON_PREFILL_TILE64_POLICY="$tile_policy" \
    VLLM_GDN_CAUSAL_CONV1D_BLOCK_M="$gdn_block" \
    VLLM_GDN_CHUNK_SIZE="$gdn_chunk" \
    FLA_GDN_FIX_BT="$fla_fix" \
    FLA_TRIL_PRECISION="$tril_precision" \
    VLLM_GFX936_FUSED_GATE_UP_SILU="$llmm1silu" \
    RANGE="$range" \
    NUM_PROMPTS="$num_prompts" \
    OUT="$case_out" \
    START_VLLM=1 \
    STOP_EXISTING_VLLM=1 \
    WAIT_HEALTH_TIMEOUT_S="$WAIT_HEALTH_TIMEOUT_S" \
        bash "$GOVINDA_DIR/tools/codex_run_p0_smoke_once.sh" \
        2>&1 | tee "$case_out/run.log"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "$status" == "0" ]]; then
        status_text="ok"
    else
        status_text="fail:$status"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$label" "$ref" "$tile_policy" "$gdn_block" "$gdn_chunk" "$fla_fix" "$tril_precision" "$llmm1silu" \
        "$range" "$num_prompts" "$status_text" "$case_out" >> "$summary"

    if [[ "$status" != "0" ]]; then
        echo "case $label failed with status $status; continuing to next case"
    fi
done

echo
echo "=== summary ==="
cat "$summary"
