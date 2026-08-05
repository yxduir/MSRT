import os.path as osp
import random
import json, yaml
import copy
import os
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
import multiprocessing as mp

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    import datasets as _datasets
except ImportError:
    _datasets = None

iso3_to_iso2_map = {
    "afr": "af", "amh": "am", "ara": "ar", "asm": "as", "ast": "ast",
    "azj": "az", "bel": "be", "bul": "bg", "ben": "bn", "bos": "bs",
    "cat": "ca", "ceb": "ceb", "ckb": "ckb", "cmn": "zh", "ces": "cs",
    "cym": "cy", "dan": "da", "deu": "de", "ell": "el", "eng": "en",
    "spa": "es", "est": "et", "fas": "fa", "ful": "ff", "fin": "fi",
    "tgl": "tl", "fra": "fr", "gle": "ga", "glg": "gl", "guj": "gu",
    "hau": "ha", "heb": "he", "hin": "hi", "hrv": "hr", "hun": "hu",
    "hye": "hy", "ind": "id", "ibo": "ig", "isl": "is", "ita": "it",
    "jpn": "ja", "jav": "jv", "kat": "ka", "kam": "kam", "kea": "kea",
    "kaz": "kk", "khm": "km", "kan": "kn", "kor": "ko", "kir": "ky",
    "ltz": "lb", "lug": "lg", "lin": "ln", "lao": "lo", "lit": "lt",
    "luo": "luo", "lav": "lv", "mri": "mi", "mkd": "mk", "mal": "ml",
    "mon": "mn", "mar": "mr", "msa": "ms", "mlt": "mt", "mya": "my",
    "nob": "nb", "npi": "ne", "nld": "nl", "nso": "nso", "nya": "ny",
    "oci": "oc", "orm": "om", "ory": "or", "pan": "pa", "pol": "pl",
    "pus": "ps", "por": "pt", "ron": "ro", "rus": "ru", "snd": "sd",
    "slk": "sk", "slv": "sl", "sna": "sn", "som": "so", "srp": "sr",
    "swe": "sv", "swh": "sw", "tam": "ta", "tel": "te", "tgk": "tg",
    "tha": "th", "tur": "tr", "ukr": "uk", "umb": "umb", "urd": "ur",
    "uzb": "uz", "vie": "vi", "wol": "wo", "xho": "xh", "yor": "yo",
    "yue": "yue", "zul": "zu"
}

# --- 模块级函数，供 datasets.map / multiprocessing 多进程调用 ---
def _load_and_mel(args):
    """多进程 worker（兼容旧接口）：加载音频 → pad_or_trim → log_mel_spectrogram"""
    audio_path, mel_size = args
    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        audio_mel = whisper.log_mel_spectrogram(audio, n_mels=mel_size)
        return audio_path, audio_mel
    except Exception as e:
        print(f"  [wav_cache warn] 加载失败: {audio_path}, {e}")
        return audio_path, None

class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()

        def match_source(source, src_lang, tgt_lang):
            """
            解析 source 模式，支持格式: "eng*28", "eng*ab", "12*28"
            """
            if '*' not in source:
                return False
            
            parts = source.split('*')
            if len(parts) != 2:
                return False
                
            src_pattern, tgt_pattern = parts

            # 内部辅助逻辑：优先从 groups 找，找不到则视为单个语言代码
            def get_langs(pattern):
                # 兼容 "1" 或 "language_1" 两种写法
                key = pattern.replace("language_", "")
                return groups.get(key, [pattern])

            src_langs = get_langs(src_pattern)
            tgt_langs = get_langs(tgt_pattern)

            return src_lang in src_langs and tgt_lang in tgt_langs
        
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.mode = dataset_config.get("mode", "srt")
        self.use_template = dataset_config.get("use_template", False)
                
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", "")
        self.bf16 = dataset_config.get("bf16", True)
        self.fp16 = dataset_config.get("fp16", False)
        self.mel_size = dataset_config.get("mel_size", 128) # 80 for whisper large v1 and v2, 128 for large v3
        self.source = dataset_config.get("source", "eng")
        self.lang_code = dataset_config.get("lang_code", "eng")
        self.lang_max_train = dataset_config.get("lang_max_train", 50000)
        self.lang_max_val = dataset_config.get("lang_max_val", 20)

        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", 80)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.validnum = dataset_config.get("validnum", -1)
        self.input_type = dataset_config.get("input_type", "mel")
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"
        self.data_dir = os.path.dirname(dataset_config.get("val_data_path"))+"/"
        self.continue_data_path = dataset_config.get("continue_data_path", None)
        self.wav_cache = dataset_config.get("wav_cache", False)

        groups = {
                "01": ['eng'],
                "02": ['cmn', 'eng'],
                "45": [
                    'ara', 'azj', 'bul', 'ben', 'cat', 'ces', 'dan', 'deu', 'ell', 'eng',
                    'spa', 'fas', 'fin', 'fra', 'heb', 'hin', 'hrv', 'hun', 'ind', 'ita',
                    'jpn', 'kaz', 'khm', 'kor', 'lao', 'msa', 'mya', 'nob', 'nld', 'pol',
                    'por', 'ron', 'rus', 'slk', 'slv', 'swe', 'tam', 'tha', 'tgl', 'tur',
                    'urd', 'uzb', 'vie', 'yue', 'cmn'],
                "70": ['afr', 'amh', 'ara', 'asm', 'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ces', 'cmn', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'fas', 'fin', 'fra', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kan', 'kat', 'kaz', 'khm', 'kir', 'kor', 'lao', 'lav', 'lit', 'mal', 'mkd', 'mon', 'msa', 'mya', 'nld', 'nob', 'npi', 'pan', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'ukr', 'urd', 'uzb', 'vie', 'yue'],
                "apec": ['eng', 'msa', 'fra', 'spa', 'cmn', 'yue', 'ind', 'jpn', 'kor', 'tgl', 'rus', 'tha', 'vie', 'tam']
            }
        
        # 设置随机种子，确保结果可复现
        random_seed = 42  # 可以替换为任意整数
        random.seed(random_seed)

        # 加载已处理的 id 集合，用于跳过重复数据
        self.processed_ids = set()
        if self.continue_data_path is not None and os.path.exists(self.continue_data_path):
            print(f"读取已处理数据: {self.continue_data_path}")
            with open(self.continue_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if 'id' in data:
                            self.processed_ids.add(data['id'])
            print(f"已处理 id 数量: {len(self.processed_ids)}")

        self.data_list = []
        self.count = 0
        src_lang_counts = {}  # 结构将变为: {("cmn", "eng"): count, ("eng", "kor"): count, ...}

        if split == "train":
            categorized_data = {}  # 临时桶格式: { ("src", "tgt"): [data1, data2, ...], ... }
            with open(dataset_config.get("train_data_path"), encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data_id = data_dict.get("id")
                    if data_id in self.processed_ids:
                        continue
                    src_lang = data_dict["src"]
                    tgt_lang = data_dict["tgt"]
                    
                    # 1. 过滤：只要匹配 match_source，就按 (src, tgt) 组合存入对应的桶里
                    if match_source(self.source, src_lang, tgt_lang):
                        lang_pair = (src_lang, tgt_lang)  # 使用元组作为字典的键
                        if lang_pair not in categorized_data:
                            categorized_data[lang_pair] = []
                        categorized_data[lang_pair].append(data_dict)

            # 2. 抽取：对每个“源语言-目标语言”组合的数据进行全局随机打乱，并截取上限
            for lang_pair, items in categorized_data.items():
                # 随机打乱当前组合下的所有数据
                random.shuffle(items)
                
                # 每个组合独立截取最多 self.lang_max_train 条数据
                selected_items = items[:self.lang_max_train]
                
                # 合并到最终的训练数据列表中
                self.data_list.extend(selected_items)
                
                # 记录该组合最终抽取的实际数量
                src_lang_counts[lang_pair] = len(selected_items)
            
            # 3. 打乱最终合并后的总列表，避免相同语种对在训练时扎堆
            random.shuffle(self.data_list) 
        else:
            # === 修正后的 val 分支逻辑 ===
            categorized_data = {}  # 验证集的临时桶
            with open(dataset_config.get("val_data_path"), encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    data_id = data_dict.get("id")
                    if data_id in self.processed_ids:
                        continue
                    src_lang = data_dict["src"]
                    tgt_lang = data_dict["tgt"]
                    
                    # 1. 过滤并按语种对分桶
                    if match_source(self.source, src_lang, tgt_lang):
                        lang_pair = (src_lang, tgt_lang)
                        if lang_pair not in categorized_data:
                            categorized_data[lang_pair] = []
                        categorized_data[lang_pair].append(data_dict)

            # 2. 抽取：每个组合独立截取最多 self.lang_max_val 条数据
            lang_max_val = getattr(self, "lang_max_val", self.lang_max_train) 
            
            for lang_pair, items in categorized_data.items():
                random.shuffle(items)
                selected_items = items[:lang_max_val]
                self.data_list.extend(selected_items)
                src_lang_counts[lang_pair] = len(selected_items)
                
            # 3. 打乱最终合并后的验证集总列表
            random.shuffle(self.data_list)
               
        self.printed = False  # 标志位，控制print只执行一次
        print(split, len(self.data_list))

        # --- wav_cache: 延迟加载模式，类似 test_inference_npu.py 的思路 ---
        # 不在初始化时预计算所有 mel，而是在 __getitem__ 首次访问音频时才计算并缓存
        self.audio_cache = {}
        if self.wav_cache:
            print("  [wav_cache] 启用懒加载 mel 缓存模式")
    
    def _prepare_audio(self, audio_input):
        """复刻你原有的音频加载逻辑"""
        if isinstance(audio_input, str):
            # whisper.load_audio 默认会重采样到 16kHz
            return whisper.load_audio(audio_input)
        
        if isinstance(audio_input, dict):
            audio_array = audio_input["array"]
            sr = audio_input["sampling_rate"]
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            return audio_array.astype(np.float32)
        raise ValueError("输入应为文件路径或 HF Audio 字典")

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]

        audio_key = data_dict["audio"]
        if not audio_key.startswith('/'):
            audio_path = self.data_dir + audio_key
        else:
            audio_path = audio_key
        
        src = data_dict.get("src","")
        tgt = data_dict.get("tgt","")
        asr = data_dict.get("asr","")
        s2tt = data_dict.get("s2tt","")
        source = data_dict.get("source")
        id = data_dict.get("id")

        if self.mode == "smt":
            prompt = target.split(prompt)[0]+prompt
            target = target.split(prompt)[1]
        if self.mode == "smtsrt":
            if random.random() < 0.5:  # 50%概率触发
                prompt = target.split(prompt)[0] + prompt
                target = target.split(prompt)[1]
        if self.mode == "st":
            prompt = prompt
            target = target.split(prompt)[1]
        if self.mode == "asr":
            prompt = f"<|{src}|>"
            target = asr
        if self.mode == "srt":
            prompt = f"<|{src}|><|{tgt}|>"
            target = asr+prompt+s2tt
        if self.mode == "mix":
            random_num = random.random()
            prompt = f"<|{src}|><|{tgt}|>"
            target = asr+prompt+s2tt
            if random_num < 0.1:  
                prompt = asr + prompt
                target = s2tt
            if random_num >= 0.1 and random_num <0.3:
                prompt = f"<|{src}|>"
                target = asr
        
        if self.use_template:
            prompt = f"<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_length = len(prompt_ids)
        
        if not self.printed:  # 如果没有打印过，则打印一次
            print(prompt,prompt_ids)
            print(target)
            self.printed = True  # 设置标志位，表示已经打印过了

        key = data_dict.get("key", str(index))
        
        # 音频加载: wav_cache 开启时延迟加载（类似 test_inference_npu.py），否则实时加载
        if self.wav_cache:
            if audio_path not in self.audio_cache:
                audio = self._prepare_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                self.audio_cache[audio_path] = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size)
            audio_mel = self.audio_cache[audio_path]
        else:
            audio = self._prepare_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            audio_mel = whisper.log_mel_spectrogram(audio, n_mels=self.mel_size)
        
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        
        audio_pseudo = torch.full((audio_length,), -1) # placeholder
        
        if self.inference_mode:
            audio_mel = audio_mel.to(torch.bfloat16)
        
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "audio_path": audio_path, # 修正了未定义的 audio_jsonl 变量
                "prompt_id": prompt_ids,
                "prompt_length": prompt_length,
                "source": source,
                "id": id,
                "src": src,
                "tgt": tgt,
            }
        
        if self.bf16:
            audio_mel = audio_mel.to(torch.bfloat16)
        answer = self.answer_template.format(target)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.

        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64)

        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples] #[120, 48, 82, 42]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]  #[0, 0, 0, 0]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        if self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]
            audio_paths = [s['audio_path'] for s in samples]
            ids = [s['id'] for s in samples]
            srcs = [s.get('src', '') for s in samples]
            tgts = [s.get('tgt', '') for s in samples]
            prompt_lengths = [s['prompt_length'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
                "audio_paths": audio_paths,
                "ids": ids,
                "srcs": srcs,
                "tgts": tgts,
                "prompt_lengths": prompt_lengths,
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "modality_mask": modality_mask
        }

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset