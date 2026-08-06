#!/bin/bash
# ============================================================================
# MSRT inference launcher (Ascend NPU + Accelerate + DeepSpeed)
#
# Distributed inference on a single 8-card Ascend NPU node.
# Run from the repo root:
#     bash infer_moe_npu.sh
# ============================================================================

set -e

# ---------------------------------------------------------------------------
# NPU environment variables
# ---------------------------------------------------------------------------
export MSRT_BACKEND=npu
export ALLOW_INTERNAL_FORMAT=1
export TORCH_NPU_FUSION_OP=1
export ASCEND_RT_VISIBLE_DEVICES=0

export NCCL_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

# ---------------------------------------------------------------------------
# <<< EDIT THESE: data & model paths >>>
# ---------------------------------------------------------------------------
DATA_PATH="./MSRT-4B/data/srt_test_1.jsonl"   # JSONL, one sample per line

# Model components (default to the local models/ backup in this repo)
ENCODER_HIGH_PATH="./MSRT-4B/whisper-large-v3-encoder"       # High-resource Whisper encoder
ENCODER_MID_LOW_PATH="./MSRT-4B/whisper_moe_expert" # Merged mid/low-resource encoder
TOKENIZER_PATH="./MSRT-4B/MSRT-4B-LLM"             # LLM tokenizer directory
LLM_PATH="./MSRT-4B/MSRT-4B-LLM"                   
LLM_DIM=2560
QUERY_LEN=80

# Trained checkpoint saved by the training script (model.pt)
CHECKPOINT="./MSRT-4B/msrt_4b.pt"

# ---------------------------------------------------------------------------
# Inference options
# ---------------------------------------------------------------------------
MODE="srt"                        # srt (ASR+S2TT)
MAX_NEW_TOKENS=400
BATCH_SIZE=256                    # per-device batch size
NUM_BEAMS=1
SOURCE="45*45"                        # data filter, format "src*tgt" e.g. "01*45"
DATE=$(date +%Y%m%d)               # e.g. 20260720
OUTPUT_PATH="./output/${DATE}/${SOURCE}_beam${NUM_BEAMS}_4b_moe.jsonl"
mkdir -p "$(dirname "${OUTPUT_PATH}")"  # ensure output directory exists
RESUME=true                       # resume: skip already-processed samples
# RESUME_FROM="./output/other_result.jsonl"  # (optional) source of processed IDs
WAV_CACHE=true                    # cache audio mel by filename to avoid re-reading audio

# ---------------------------------------------------------------------------
# LLM LoRA flags (仅 LLM 使用；Encoder 的 LoRA 已合并进权重)
# ---------------------------------------------------------------------------
USE_LORA=true                    # LLM LoRA (not used in the released checkpoints)
LORA_RANK=16
LORA_ALPHA=32

# Encoder mid/low LoRA：可选。当前发布权重已合并进 encoder，故默认 false；
# 若使用未合并的 encoder 权重，改为 true 启用。
USE_ENCODER_MID_LOW_LORA=false
ENC_MID_LOW_LORA_RANK=16
ENC_MID_LOW_LORA_ALPHA=32

# Assemble LoRA flags
LORA_FLAGS=""
if [ "${USE_LORA}" = "true" ]; then
    LORA_FLAGS="${LORA_FLAGS} --use_lora"
fi
if [ "${USE_ENCODER_MID_LOW_LORA}" = "true" ]; then
    LORA_FLAGS="${LORA_FLAGS} --use_encoder_mid_low_lora"
fi

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "============================================"
echo " MSRT MoE Inference (Accelerate + DeepSpeed)"
echo "============================================"
echo " Data:       ${DATA_PATH}"
echo " Output:     ${OUTPUT_PATH}"
echo " Checkpoint: ${CHECKPOINT}"
echo " Mode:       ${MODE}"
echo " Source:     ${SOURCE}"
echo " Batch:      ${BATCH_SIZE}"
echo " LoRA:       ${LORA_FLAGS}"
echo " Wav Cache:  ${WAV_CACHE}"
echo " Resume:     ${RESUME}"
echo " ResumeFrom: ${RESUME_FROM:-<default: OUTPUT_PATH>}"
echo "============================================"

accelerate launch \
    --config_file ./acceleate_config.yaml \
    infer_moe.py \
    --checkpoint "${CHECKPOINT}" \
    --encoder_high_path "${ENCODER_HIGH_PATH}" \
    --encoder_mid_low_path "${ENCODER_MID_LOW_PATH}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --llm_path "${LLM_PATH}" \
    --llm_dim ${LLM_DIM} \
    --query_len ${QUERY_LEN} \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --mode "${MODE}" \
    --source "${SOURCE}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --val_batch_size ${BATCH_SIZE} \
    --num_beams ${NUM_BEAMS} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --encoder_mid_low_lora_rank ${ENC_MID_LOW_LORA_RANK} \
    --encoder_mid_low_lora_alpha ${ENC_MID_LOW_LORA_ALPHA} \
    ${LORA_FLAGS} \
    $(if [ "${WAV_CACHE}" = "true" ]; then echo "--wav_cache"; fi) \
    $(if [ "${RESUME}" = "true" ]; then echo "--resume"; fi) \
    $(if [ -n "${RESUME_FROM:-}" ]; then echo "--resume_from" "${RESUME_FROM}"; fi)

echo ""
echo "Done! Results saved to: ${OUTPUT_PATH}"
