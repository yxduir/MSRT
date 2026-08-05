<p align="center">
  <h1 align="center">MSRT / MoE</h1>
  <p align="center">
    多语种语音大模型推理仓库 — 45 语种 <b>ASR</b>（语音识别）与 <b>S2TT</b>（语音翻译）
  </p>
</p>

> 本仓库用 **Accelerate** 统一算力（NPU / GPU），实现了框架与算力无关；模型在 **Ascend 910C** 上训练，权重同样支持在 GPU 上训练与推理。

---

## 一、环境准备

依赖管理使用 [uv](https://github.com/astral-sh/uv)（需已安装 `uv`），Python 版本 3.12。

```bash
# 1. 在仓库根目录创建虚拟环境
uv venv --python 3.12

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 下载模型权重
hf download yxdu/MSRT-4B --local-dir MSRT-4B
```

> `hf` 是 Hugging Face 官方命令行工具（`huggingface_hub`）。若提示 `hf: command not found`，先执行 `pip install -U huggingface_hub`。

---

## 二、模型推理

### 1. 安装依赖

```bash
#GPU版本
uv pip install -r requirements_gpu.txt

#NPU版本
#uv pip install -r requirements_npu.txt
```

### 2. 运行

```bash
bash infer_moe_gpu.sh

#bash infer_moe_npu.sh
```

脚本用 `accelerate launch --config_file ./acceleate_config.yaml` 启动bf推理

---

## 数据格式

输入为 JSONL（UTF-8，每行一个 JSON 对象）：

```json
{"id": "ara_azj_0", "prompt": "<|ara|><|azj|>", "asr": "<参考转写>", "s2tt": "<参考译文>",
 "audio": "ar_eg/audio/test/8063018510254830531.wav", "src": "ara", "tgt": "azj", "source": "fleurs_ara_azj"}
```

| 字段               | 说明                                                   |
| ------------------ | ------------------------------------------------------ |
| `id`             | 样本唯一标识（用于断点续传去重）                       |
| `audio`          | 音频路径，相对 JSONL 所在目录或绝对路径                |
| `src` / `tgt`  | 源 / 目标语种代码（ISO-639-3），`src` 需在支持列表内 |
| `asr` / `s2tt` | 参考转写 / 参考译文（仅用于评测指标）                  |
| `prompt`         | 信息性字段；实际 prompt 由 `src`/`tgt` 重建        |

支持语种（45）：`ara, azj, bul, ben, cat, ces, dan, deu, ell, eng, spa, fas, fin, fra, heb, hin, hrv, hun, ind, ita, jpn, kaz, khm, kor, lao, msa, mya, nob, nld, pol, por, ron, rus, slk, slv, swe, tam, tha, tgl, tur, urd, uzb, vie, yue, cmn`

## 输出格式

结果逐条写入 `--output_path` 指定的 JSONL：

```json
{"id": "...", "audio": "...", "prompt": "<|ara|><|azj|>", "src": "ara", "tgt": "azj",
 "asr": "<参考转写>", "s2tt": "<参考译文>",
 "asr_r": "<识别文本>", "s2tt_r": "<翻译文本>",
 "response": "<模型原始输出>", "time": "2026-08-05 10:00:00"}
```

- `asr_r` / `s2tt_r` 由原始 `response` 按分隔符 `<|src|><|tgt|>` 拆分得到。
