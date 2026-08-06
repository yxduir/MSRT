<p align="center">
  <h1 align="center">MSRT</h1>
  <p align="center">
    该论文提出一种资源感知的混合语音编码器（Resource-Aware Mixture of Speech Encoders），用于打破多语言语音到文本多对多翻译中的“多语种诅咒”，覆盖45种语言、共 1980 个翻译方向（45×44）。
  </p>
</p>

> 本仓库基于 Hugging Face **Accelerate** 构建统一的推理框架，兼容 NPU 与 GPU 两种后端，框架实现与具体算力解耦；模型权重基于 **Ascend 910C** 训练，亦可直接迁移至 NVIDIA GPU 进行推理与训练。

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

> **显存要求**：推理所需**最小显存为 16 GB**（GPU）。

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

---

## 引用 / Citation

```bibtex
@misc{du2026breakingcurseofmultilingualityinmanytomany,
      title={Breaking the Curse ofMultilinguality inMany-to-Many Speech-to-Text Translation via a Resource-AwareMixture of Speech Encoders}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Bo Yang and Chengpeng Fu and Yu Wang and Ming Liu},
      year={2026},
      eprint={2608.04586},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.04586}, 
}
```
