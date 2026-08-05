#!/usr/bin/env python3
"""
复用 st_dataset.py SpeechDatasetJsonl 做数据加载和 batch 构造，
使用 accelerate + deepspeed 做分布式推理。

输出 JSONL，每行包含: id, audio, prompt, src, tgt, asr, s2tt, asr_r, s2tt_r, response, time
"""

import os
import sys

# ---------------------------------------------------------------------------
# 后端自适应：NPU (默认) 或 CUDA GPU。
#   - 环境变量 MSRT_BACKEND=npu|cuda|auto 强制指定后端；
#     auto 时优先尝试 NPU（设置 NPU 环境变量并导入 torch_npu），
#     若 torch_npu 未安装则回退为 CUDA（跳过 NPU 相关设置）。
#   - GPU 运行请用 infer_moe_gpu.sh（已设 MSRT_BACKEND=cuda），
#     依赖见 requirements_gpu.txt。
# 注意：NPU 相关设置必须在 import torch 之前完成。
# ---------------------------------------------------------------------------
_BACKEND = os.environ.get("MSRT_BACKEND", "auto")

if _BACKEND == "npu":
    os.environ["ALLOW_INTERNAL_FORMAT"] = "1"
    os.environ["TORCH_NPU_FUSION_OP"] = "1"
    import torch_npu  # noqa: F401
elif _BACKEND == "cuda":
    pass  # CUDA 后端：不设置 NPU 环境变量，也不导入 torch_npu
else:  # auto：优先 NPU，导入失败则回退 CUDA
    os.environ.setdefault("ALLOW_INTERNAL_FORMAT", "1")
    os.environ.setdefault("TORCH_NPU_FUSION_OP", "1")
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass

import torch
import torch.distributed as dist
import argparse
import json
import time
from tqdm import tqdm

from accelerate import Accelerator
from transformers import set_seed

from st_dataset import get_speech_dataset
from srt_moe import MoeConfig, Moe, lang_to_expert_map


# ---------------------------------------------------------------------------
# 解析模型输出，提取 asr_r 和 s2tt_r
# ---------------------------------------------------------------------------
def parse_response(response: str, src: str, tgt: str):
    """从模型输出中解析 ASR 识别文本和 S2TT 翻译文本"""
    asr_r = ""
    s2tt_r = ""
    sep = f"<|{src}|><|{tgt}|>"
    if sep in response:
        parts = response.split(sep, 1)
        asr_r = parts[0].strip()
        s2tt_r = parts[1].strip() if len(parts) > 1 else ""
    else:
        sep_src = f"<|{src}|>"
        if sep_src in response:
            parts = response.split(sep_src, 1)
            asr_r = parts[0].strip()
            s2tt_r = parts[1].strip() if len(parts) > 1 else ""
        else:
            asr_r = response
    return asr_r, s2tt_r


# ---------------------------------------------------------------------------
# 主推理逻辑
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MSRT MoE Inference (NPU + Accelerate + DeepSpeed)")

    # --- 模型加载 ---
    parser.add_argument("--model_path", type=str, default=None,
                        help="save_pretrained 合并后的模型目录路径 (优先使用)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="训练 checkpoint 路径 (model.pt)，与各组件路径配合使用")

    # --- 各组件路径 (checkpoint 模式使用) ---
    parser.add_argument("--encoder_high_path", type=str,
                        default="/home/ma-user/work/models/whisper-large-v3-encoder")
    parser.add_argument("--encoder_mid_low_path", type=str,
                        default="/home/ma-user/work/huggingface/merged_encoder_mid_low_lora")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/home/ma-user/work/models/MiLMMT-46-4B-text-add")
    parser.add_argument("--llm_path", type=str,
                        default="/home/ma-user/work/models/MiLMMT-46-4B-text-add")
    parser.add_argument("--llm_dim", type=int, default=2560)
    parser.add_argument("--query_len", type=int, default=80)
    parser.add_argument("--fallback_expert", type=int, default=0)

    # --- 推理参数 ---
    parser.add_argument("--data_path", type=str, required=True,
                        help="推理数据 JSONL 路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出 JSONL 路径")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="启用断点续传：增量写入结果，跳过已处理样本")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="断点续传的参考文件路径（已处理样本 ID 来源）。未指定时默认使用 --output_path")
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help="推理 batch size")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--mode", type=str, default="srt",
                        choices=["srt", "asr"])
    parser.add_argument("--source", type=str, default="*",
                        help="数据筛选，格式: src*tgt，如 \"45*45\" 或 \"eng*28\"")
    parser.add_argument("--lang_max_val", type=int, default=999999,
                        help="每种语言对最多推理多少条 (默认=全量)")
    parser.add_argument("--wav_cache", action="store_true", default=False,
                        help="如果开启，直接用 wav 文件名映射 audio_mel，避免重复读取音频")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # --- LoRA 配置 (与训练保持一致) ---
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="LLM 是否使用 LoRA")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_encoder_mid_low_lora", action="store_true", default=False,
                        help="中低资源 Encoder 是否使用 LoRA（当前发布权重已合并，默认关闭）")
    parser.add_argument("--encoder_mid_low_lora_rank", type=int, default=16)
    parser.add_argument("--encoder_mid_low_lora_alpha", type=int, default=32)

    # --- Accelerate / DeepSpeed ---
    parser.add_argument("--project_dir", type=str, default="logs")
    parser.add_argument("--with_tracking", action="store_true", default=False)

    args = parser.parse_args()

    # ================================================================
    # 0. 断点续传：读取已处理样本 ID
    # ================================================================
    processed_ids = set()
    if args.resume:
        resume_from = args.resume_from if args.resume_from is not None else args.output_path
    else:
        resume_from = None

    resume_mode = resume_from is not None and os.path.exists(resume_from)
    if resume_mode:
        print(f"[Resume] 读取已处理数据: {resume_from}")
        with open(resume_from, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            processed_ids.add(data['id'])
                    except json.JSONDecodeError:
                        continue
        print(f"[Resume] 已处理样本数: {len(processed_ids)}")

    # ================================================================
    # 1. 初始化 Accelerator
    # ================================================================
    if args.with_tracking:
        accelerator = Accelerator(
            log_with="aim",
            gradient_accumulation_steps=1,
            project_dir=args.project_dir,
        )
    else:
        accelerator = Accelerator()

    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Accelerator device: {device}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Num processes: {accelerator.num_processes}")

    set_seed(42)

    # 断点续传信息同步到所有进程
    if resume_mode and accelerator.is_main_process:
        print(f"[Resume] 断点续传模式启用，将跳过 {len(processed_ids)} 个已处理样本")

    # ================================================================
    # 2. 加载模型 (完全对齐 hf_train_moe.py)
    # ================================================================
    if accelerator.is_main_process:
        print("=" * 60)
        print("Loading model...")

    if args.model_path is not None:
        if accelerator.is_main_process:
            print(f"  From pretrained: {args.model_path}")
        model = Moe.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
    elif args.checkpoint is not None:
        if accelerator.is_main_process:
            print(f"  Encoder High:     {args.encoder_high_path}")
            print(f"  Encoder Mid/Low:  {args.encoder_mid_low_path}")
            print(f"  LLM:              {args.llm_path}")
            print(f"  Tokenizer:        {args.tokenizer_path}")
            print(f"  Checkpoint:       {args.checkpoint}")

        config = MoeConfig(
            encoder_high_path=args.encoder_high_path,
            encoder_mid_low_path=args.encoder_mid_low_path,
            llm_path=args.llm_path,
            tokenizer_path=args.tokenizer_path,
            llm_dim=args.llm_dim,
            query_len=args.query_len,
            lang_to_expert_map=lang_to_expert_map,
            fallback_expert=args.fallback_expert,
        )
        model = Moe(config)

        # --- 对齐训练: 先 apply LoRA, 再 load checkpoint ---
        if args.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            peft_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model.llm = get_peft_model(model.llm, peft_config)
            if accelerator.is_main_process:
                print("  LLM LoRA applied")

        # Encoder (mid/low) 的 LoRA：可选。当前发布权重已合并进 encoder，
        # 因此默认关闭；若使用未合并的 encoder 权重，可开启此选项。
        if args.use_encoder_mid_low_lora:
            from peft import LoraConfig, get_peft_model
            encoder_lora_config = LoraConfig(
                r=args.encoder_mid_low_lora_rank,
                lora_alpha=args.encoder_mid_low_lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model.moe_encoder.encoder_mid_low = get_peft_model(
                model.moe_encoder.encoder_mid_low, encoder_lora_config
            )
            if accelerator.is_main_process:
                print("  Encoder Mid/Low LoRA applied")

        # 加载 checkpoint 权重
        if accelerator.is_main_process:
            print(f"  Loading checkpoint weights from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            ckpt_total = len(state_dict)
            loaded_keys = ckpt_total - len(unexpected_keys)
            model_total = len(model.state_dict())
            filled_keys = model_total - len(missing_keys)
            print(f"  Checkpoint: {ckpt_total} keys, loaded {loaded_keys} "
                  f"({filled_keys}/{model_total} model keys filled)")

            # 诊断 LoRA 加载情况
            ckpt_lora_keys = {k for k in state_dict if "lora_" in k}
            model_lora_keys = {k for k in model.state_dict() if "lora_" in k}
            loaded_lora = ckpt_lora_keys - set(unexpected_keys)
            missing_lora = ckpt_lora_keys & set(unexpected_keys)
            print(f"  LoRA: checkpoint has {len(ckpt_lora_keys)}, "
                  f"model has {len(model_lora_keys)}, "
                  f"loaded {len(loaded_lora)}, "
                  f"missed {len(missing_lora)}")
            if missing_lora:
                print(f"  ⚠️ LoRA keys NOT loaded (checkpoint has but model missing):")
                for k in sorted(missing_lora)[:10]:
                    print(f"      {k}")
                if len(missing_lora) > 10:
                    print(f"      ... and {len(missing_lora) - 10} more")
            if len(ckpt_lora_keys) == 0:
                print(f"  ⚠️ Checkpoint contains NO lora_ keys. Did training use --use_lora?")

            if missing_keys:
                # 按模块统计
                missing_by_module = {}
                for k in missing_keys:
                    mod = k.split(".")[0] if "." in k else k
                    missing_by_module.setdefault(mod, 0)
                    missing_by_module[mod] += 1
                print(f"  Missing keys breakdown:")
                for mod, cnt in sorted(missing_by_module.items()):
                    print(f"    - {mod}: {cnt}")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)} (ignored)")
    else:
        if accelerator.is_main_process:
            print("ERROR: 必须指定 --model_path 或 --checkpoint")
        sys.exit(1)

    model.eval()
    model = model.to(device)  
    tokenizer = model.tokenizer  # 复用模型内部的 tokenizer

    if accelerator.is_main_process:
        print(f"  Model loaded. EOS={tokenizer.eos_token_id}, "
              f"BOS={tokenizer.bos_token_id}, PAD={tokenizer.pad_token_id}")
        print("=" * 60)

    # ================================================================
    # 3. 加载数据 (复用 st_dataset.py)
    # ================================================================
    dataset_config = {
        'train_data_path': args.data_path,
        'val_data_path': args.data_path,
        'train_split': 'train',
        'test_split': 'test',
        'source': args.source,
        'mode': args.mode,
        'fix_length_audio': args.query_len,
        'use_template': False,
        'inference_mode': True,
        'lang_max_train': args.lang_max_val,
        'lang_max_val': args.lang_max_val,
        'wav_cache': args.wav_cache,
        'continue_data_path': resume_from,
    }
    dataset = get_speech_dataset(tokenizer=tokenizer, dataset_config=dataset_config, split="val")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        pin_memory=True,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=dataset.collator,
    )

    if accelerator.is_main_process:
        print(f"Dataset: {len(dataset)} samples (source filter: '{args.source}')")
        print(f"Batches: {len(dataloader)} (batch_size={args.val_batch_size})")

    # ================================================================
    # 4. Prepare model + dataloader with accelerator (对齐训练)
    # ================================================================
    model, dataloader = accelerator.prepare(model, dataloader)

    # ================================================================
    # 5. 推理循环（断点续传：每 batch 增量写入）
    # ================================================================
    total_processed = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Inference",
        disable=not accelerator.is_main_process,
    )

    for batch_idx, batch in pbar:
        batch_ids = batch["ids"]
        batch_targets = batch["targets"]
        batch_srcs = batch["srcs"]
        batch_tgts = batch["tgts"]
        batch_audios = batch["audio_paths"]

        # 当前 batch 的结果
        batch_results = []

        # 获取 unwrapped model 做 generate
        unwrapped = accelerator.unwrap_model(model)

        try:
            with torch.no_grad():
                batch_output_ids = unwrapped.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_mel=batch["audio_mel"],
                    modality_mask=batch["modality_mask"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
        except Exception as e:
            if accelerator.is_main_process:
                print(f"\n[ERROR] Batch {batch_idx} failed: {e}")
            for i in range(len(batch_ids)):
                asr_gt_i, s2tt_gt_i = parse_response(batch_targets[i], batch_srcs[i], batch_tgts[i])
                batch_results.append({
                    "id": batch_ids[i],
                    "audio": batch_audios[i],
                    "prompt": f"<|{batch_srcs[i]}|><|{batch_tgts[i]}|>",
                    "src": batch_srcs[i],
                    "tgt": batch_tgts[i],
                    "asr": asr_gt_i,
                    "s2tt": s2tt_gt_i,
                    "asr_r": "", "s2tt_r": "", "response": "",
                    "error": str(e),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                })
        else:
            # 逐条解码
            for i in range(len(batch_ids)):
                output_ids = batch_output_ids[i]
                # HF generate + inputs_embeds 的可能返回:
                #   A) [input_prefix_pad..., generated_tokens...]
                #   B) [generated_tokens...]
                # 取全部 + strip 头尾 pad/eos, 两种都能兼容
                gen_list = output_ids.tolist()
                # 去掉尾部 pad(0)/eos(1)
                while gen_list and gen_list[-1] in (0, 1):
                    gen_list.pop()
                # 去掉头部 pad(0)
                while gen_list and gen_list[0] == 0:
                    gen_list.pop(0)

                out = tokenizer.decode(gen_list, skip_special_tokens=True)
                asr_r, s2tt_r = parse_response(out, batch_srcs[i], batch_tgts[i])

                # 解析 GT (从 targets 中提取 asr 和 s2tt)
                asr_gt_i, s2tt_gt_i = parse_response(batch_targets[i], batch_srcs[i], batch_tgts[i])
                prompt_i = f"<|{batch_srcs[i]}|><|{batch_tgts[i]}|>"

                batch_results.append({
                    "id": batch_ids[i],
                    "audio": batch_audios[i],
                    "prompt": prompt_i,
                    "src": batch_srcs[i],
                    "tgt": batch_tgts[i],
                    "asr": asr_gt_i,
                    "s2tt": s2tt_gt_i,
                    "asr_r": asr_r,
                    "s2tt_r": s2tt_r,
                    "response": out,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                })

        # ----------------------------------------------------------
        # 断点续传：多卡 gather + 主进程增量写入
        # ----------------------------------------------------------
        if accelerator.num_processes > 1:
            try:
                gathered = [None for _ in range(accelerator.num_processes)]
                dist.all_gather_object(gathered, batch_results)
                if accelerator.is_main_process:
                    batch_results = [r for sublist in gathered for r in sublist]
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"\n[WARN] Batch {batch_idx} gather 失败: {e}，仅保存本地结果")
                # gather 失败时各进程写自己的结果（用 rank 后缀区分）
                rank = accelerator.process_index
                partial_path = f"{args.output_path}.partial_rank{rank}"
                with open(partial_path, 'a', encoding='utf-8') as f:
                    for r in batch_results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                continue

        # 主进程增量写入（追加模式，保证中断后已完成的 batch 不丢失）
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
            with open(args.output_path, 'a', encoding='utf-8') as f:
                for r in batch_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    f.flush()
            total_processed += len(batch_results)
            pbar.set_postfix({"saved": total_processed})

        # debug: 打印最后一条（只用 r_last，它自带 id/audio/prompt/asr/s2tt，
        # 避免多卡 gather 后 batch_ids 与 r_last 来自不同 rank 造成的错位）
        if accelerator.is_local_main_process and batch_results:
            r_last = batch_results[-1]          # 全局最后一卡的最后一条（跨卡一致）
            print(f"\n{'='*30} Batch Check {'='*30}")
            print(f"ID: {r_last['id']}")
            print(f"Audio: {r_last['audio']}")
            print(f"prompt: {r_last.get('prompt', '')}")
            print(f"     ASR GT: {r_last['asr']}")
            print(f" ASR Result: {r_last['asr_r']}")
            print(f"    S2TT GT: {r_last['s2tt']}")
            print(f"S2TT Result: {r_last['s2tt_r']}")
            print(f"OUT  Result: {r_last.get('response', '')}")
            print(f"{'='*73}\n")

    # ================================================================
    # 6. 收尾：同步 + 合并 partial 文件 + 去重
    # ================================================================
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # 合并 gather 失败时产生的 partial 文件
        import glob
        partial_files = sorted(glob.glob(f"{args.output_path}.partial_rank*"))
        if partial_files:
            print(f"合并 {len(partial_files)} 个 partial 文件...")
            with open(args.output_path, 'a', encoding='utf-8') as outf:
                for pf in partial_files:
                    with open(pf, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            if line.strip():
                                outf.write(line)
                    os.remove(pf)
                    print(f"  已合并并删除: {pf}")

        # 去重（安全处理，正常不会有重复）
        seen = set()
        deduped = []
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        rid = r.get("id", "")
                        if rid and rid not in seen:
                            seen.add(rid)
                            deduped.append(r)
                    except json.JSONDecodeError:
                        continue

        # 写回去重后的结果
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for r in deduped:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n✅ 推理完成，共 {len(deduped)} 条结果")
        print(f"   输出文件: {os.path.abspath(args.output_path)}")


if __name__ == "__main__":
    main()
