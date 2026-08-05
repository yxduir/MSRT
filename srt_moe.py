import os
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from transformers import (
    PretrainedConfig, 
    PreTrainedModel, 
    WhisperModel, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModel,
    Blip2QFormerConfig, 
    Blip2QFormerModel
)

lang_to_expert_map = {
    # ==========================================
    # Head 0: 高资源语言 (High Resource) - 15 种
    # ==========================================
    "<|ara|>": 0, "<|ben|>": 0, "<|cat|>": 0, "<|ces|>": 0, "<|deu|>": 0, 
    "<|eng|>": 0, "<|fra|>": 0, "<|ita|>": 0, "<|nld|>": 0, "<|pol|>": 0, 
    "<|ron|>": 0, "<|spa|>": 0, "<|jpn|>": 0, "<|cmn|>": 0, "<|fin|>": 0,

    # ==========================================
    # Head 1: 中低资源语言 (Medium & Low Resource) - 30 种
    # ==========================================
    # --- 中资源 (Medium) ---
    "<|vie|>": 1, "<|ind|>": 1, "<|tgl|>": 1, "<|tam|>": 1, "<|dan|>": 1, 
    "<|ell|>": 1, "<|hin|>": 1, "<|hrv|>": 1, "<|por|>": 1, "<|rus|>": 1, 
    "<|slk|>": 1, "<|urd|>": 1, "<|kor|>": 1, "<|tha|>": 1, "<|kaz|>": 1, 
    "<|tur|>": 1, "<|uzb|>": 1, "<|hun|>": 1,
    # --- 低资源 (Low) ---
    "<|heb|>": 1, "<|khm|>": 1, "<|msa|>": 1, "<|bul|>": 1, "<|fas|>": 1, 
    "<|nob|>": 1, "<|slv|>": 1, "<|swe|>": 1, "<|lao|>": 1, "<|mya|>": 1, 
    "<|yue|>": 1, "<|azj|>": 1
}

# --- 1. 定义配置类 (显式区分高资源与中低资源路径) ---
class MoeConfig(PretrainedConfig):
    model_type = "moe"
    
    def __init__(
        self,
        encoder_high_path: str = "",     # 高资源 Whisper 路径
        encoder_mid_low_path: str = "",  # 中低资源 Whisper 路径
        llm_path: str = "",
        tokenizer_path: str = "",
        llm_dim: int = 3840,
        query_len: int = 80,
        encoder_dim: int = 1280,
        qformer_layers: int = 8,
        lang_to_expert_map: Dict[str, int] = None, 
        fallback_expert: int = 0,        # 默认回退到高资源头 (Head 0)
        **kwargs
    ):
        self.encoder_high_path = encoder_high_path
        self.encoder_mid_low_path = encoder_mid_low_path
        self.llm_path = llm_path
        self.tokenizer_path = tokenizer_path
        self.llm_dim = llm_dim
        self.query_len = query_len
        self.encoder_dim = encoder_dim
        self.qformer_layers = qformer_layers
        self.lang_to_expert_map = lang_to_expert_map or {}
        self.fallback_expert = fallback_expert
        super().__init__(**kwargs)

# --- 2. 辅助组件 ---
def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    mask = pad_targets != ignore_label
    if mask.sum() == 0:
        return torch.tensor(0.0).to(pad_outputs.device)
    numerator = torch.sum(pad_outputs.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator.float() / denominator.float()) * 100 

class QFormerModule(nn.Module):
    def __init__(self, encoder_dim=1280, query_len=80, num_layers=8):
        super().__init__()
        self.query_len = query_len
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = encoder_dim
        configuration.num_hidden_layers = num_layers

        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)
        self.hidden_size = configuration.hidden_size

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        return query_output.last_hidden_state

class MLPProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=3840):
        super().__init__()
        if output_dim <= 1536:
            self.linear = nn.Linear(input_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim, eps=1e-5)
            self.mode = "single"
        else:
            mid_dim = 1536 if output_dim <= 3072 else 2560
            self.linear1 = nn.Linear(input_dim, mid_dim)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(mid_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim, eps=1e-5)
            self.mode = "double"

    def forward(self, x):
        if self.mode == "single":
            return self.norm(self.linear(x))
        else:
            return self.norm(self.linear2(self.relu(self.linear1(x))))


# --- 3. MoE 语音编码器核心组件 ---
class MoeSpeechEncoder(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        # 初始化 MoE 双专家
        self.encoder_high = WhisperModel.from_pretrained(config.encoder_high_path).encoder
        self.encoder_mid_low = WhisperModel.from_pretrained(config.encoder_mid_low_path).encoder

    def forward(self, audio_mel, expert_ids):
        """
        audio_mel: [B, M, T]
        expert_ids: [B] 一维张量，指定每个样本该去哪个头 (0: High, 1: Mid/Low)
        """
        B = audio_mel.shape[0]
        final_output = None
        
        # 1. 过滤高资源样本
        mask_high = (expert_ids == 0)
        if mask_high.any():
            idx_high = mask_high.nonzero(as_tuple=True)[0]
            out_high = self.encoder_high(audio_mel[idx_high]).last_hidden_state
            
            # 延迟初始化
            if final_output is None:
                final_output = torch.zeros(
                    B, out_high.shape[1], out_high.shape[2], 
                    device=audio_mel.device, dtype=out_high.dtype
                )
            final_output[idx_high] = out_high
            
        # 2. 过滤中低资源样本
        mask_mid_low = (expert_ids == 1)
        if mask_mid_low.any():
            idx_mid_low = mask_mid_low.nonzero(as_tuple=True)[0]
            out_mid_low = self.encoder_mid_low(audio_mel[idx_mid_low]).last_hidden_state
            
            # 如果全是中低资源，前面没有初始化过，则此时初始化
            if final_output is None:
                final_output = torch.zeros(
                    B, out_mid_low.shape[1], out_mid_low.shape[2], 
                    device=audio_mel.device, dtype=out_mid_low.dtype
                )
            final_output[idx_mid_low] = out_mid_low

        return final_output


# --- 4. 主模型类 ---
class Moe(PreTrainedModel):
    config_class = MoeConfig
    _auto_class = "AutoModel"

    def __init__(self, config: MoeConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

        # 初始化 MoE 双专家语音编码器
        self.moe_encoder = MoeSpeechEncoder(config)
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_path, 
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16
        )
        self.q_former = QFormerModule(
            encoder_dim=config.encoder_dim,
            query_len=config.query_len,
            num_layers=config.qformer_layers
        )
        self.mlp = MLPProjector(
            input_dim=self.q_former.hidden_size,
            output_dim=config.llm_dim
        )
        self.to(dtype=torch.bfloat16)
        
        # 构建 Token ID 到 Expert 的映射缓存
        self.token_id_to_expert = {}
        for lang_token, expert_idx in config.lang_to_expert_map.items():
            token_id = self.tokenizer.convert_tokens_to_ids(lang_token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                self.token_id_to_expert[token_id] = expert_idx

        self.post_init()

    def _extract_expert_ids(self, input_ids: torch.LongTensor, batch_size: int, device: torch.device):
        """扫描 input_ids，遇到属于映射表的语种 token 则分配给对应头"""
        expert_ids = torch.full((batch_size,), self.config.fallback_expert, dtype=torch.long, device=device)
        if input_ids is not None and len(self.token_id_to_expert) > 0:
            for i in range(batch_size):
                for token_id in input_ids[i].tolist():
                    if token_id in self.token_id_to_expert:
                        expert_ids[i] = self.token_id_to_expert[token_id]
                        break
        return expert_ids

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(self,
                audio_mel: torch.Tensor = None,
                audio_mel_post_mask: torch.Tensor = None,
                modality_mask: torch.Tensor = None,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                inference_mode: bool = False,
                **kwargs):
        
        # 1. 提取当前 Batch 每条数据的所属头 (0 或 1)
        batch_size = audio_mel.shape[0]
        expert_ids = self._extract_expert_ids(input_ids, batch_size, audio_mel.device)

        # 2. MoE 双专家特征提取并重组
        encoder_outs = self.moe_encoder(audio_mel, expert_ids)
        
        encoder_outs = self.q_former(encoder_outs, audio_mel_post_mask)
        encoder_outs = self.mlp(encoder_outs)

        # 3. 获取文本 Embeddings
        if input_ids is not None:
            input_ids_cleaned = input_ids.clone()
            input_ids_cleaned[input_ids_cleaned == -1] = 0
            inputs_embeds = self.llm.get_input_embeddings()(input_ids_cleaned)

        # 4. 多模态融合
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                length = int(modality_lengths[i])
                start = int(modality_mask_start_indices[i])
                encoder_outs_pad[i, start : start + length] = encoder_outs[i, :length]
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        if inference_mode:
            return inputs_embeds, attention_mask

        # 5. LLM 前向传播
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        # 6. 计算准确率
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, dim=-1)
            acc = compute_accuracy(preds[:, :-1], labels[:, 1:], ignore_label=-100)

        return model_outputs, acc

    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 400),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", False),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            no_repeat_ngram_size=5,
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return model_outputs