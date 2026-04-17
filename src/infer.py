"""封装 MLX 推理（Stage1 VLM / Stage2 LLM）。

通过环境变量 USE_MLX=1 切换 MLX 后端，否则 fallback 到 transformers。
注入 LoRA adapter 路径，前后对比使用。
"""
from __future__ import annotations
from pathlib import Path
import os
import logging
from functools import lru_cache

from . import config as C

logger = logging.getLogger(__name__)


def _use_mlx() -> bool:
    return os.environ.get("USE_MLX", "1") == "1"


def _unwrap_mlx(result):
    """兼容新旧 mlx_vlm / mlx_lm generate() 返回：新版是 GenerationResult(.text)，
    老版直接是 str。"""
    if isinstance(result, str):
        return result
    return getattr(result, "text", str(result))


# ---------- Stage 1 (VLM) ----------

def _mlx_vlm_config(model, model_path: str):
    """取 MLX-VLM 模型 config（新版 apply_chat_template 必需）。

    优先 `model.config`；否则尝试 `mlx_vlm.utils.load_config`；都没有返回空 dict。
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        return cfg
    try:
        from mlx_vlm.utils import load_config
        return load_config(model_path)
    except Exception:
        return {}


@lru_cache(maxsize=2)
def load_vlm(adapter: str | None = None):
    """懒加载 VLM。adapter=None 为基线；传路径则应用 LoRA。"""
    if _use_mlx():
        from mlx_vlm import load as mlx_load
        model, processor = mlx_load(C.STAGE1_VLM_MLX, adapter_path=adapter)
        return ("mlx", model, processor)
    else:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        import torch
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            C.STAGE1_VLM_HF, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(C.STAGE1_VLM_HF)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        return ("hf", model, processor)


def extract_table_from_image(image_path: str | Path,
                              adapter: str | None = None,
                              max_tokens: int = 1024) -> str:
    """输入图片，输出 markdown 表格文本。"""
    backend, model, processor = load_vlm(adapter)
    prompt = "请提取图中所有表格，以标准 Markdown 格式输出。如无表格输出 '无表格'。"
    if backend == "mlx":
        from mlx_vlm import generate as mlx_gen
        from mlx_vlm.prompt_utils import apply_chat_template
        cfg = _mlx_vlm_config(model, C.STAGE1_VLM_MLX)
        formatted = apply_chat_template(processor, config=cfg, prompt=prompt, num_images=1)
        result = mlx_gen(model, processor, formatted, image=[str(image_path)],
                         max_tokens=max_tokens, verbose=False)
        return _unwrap_mlx(result)
    else:
        from PIL import Image
        img = Image.open(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]


# ---------- Stage 2 (LLM) ----------

@lru_cache(maxsize=4)
def load_llm(adapter: str | None = None, model_id: str | None = None):
    """懒加载 Stage 2 LLM。

    - adapter：LoRA 适配器路径（可选）
    - model_id：覆盖默认 base 模型（可选；用于「v1 vs v2」这类同任务不同模型对比）
    """
    if _use_mlx():
        from mlx_lm import load as mlx_load
        mid = model_id or C.STAGE2_LLM_MLX
        model, tokenizer = mlx_load(mid, adapter_path=adapter)
        return ("mlx", model, tokenizer)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        mid = model_id or C.STAGE2_LLM_HF
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map="auto"
        )
        tok = AutoTokenizer.from_pretrained(mid)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        return ("hf", model, tok)


def chat(messages: list[dict], adapter: str | None = None,
         model_id: str | None = None,
         max_tokens: int = 512, temperature: float = 0.1) -> str:
    backend, model, tok = load_llm(adapter=adapter, model_id=model_id)
    if backend == "mlx":
        from mlx_lm import generate as mlx_gen
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        result = mlx_gen(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return _unwrap_mlx(result)
    else:
        import torch
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=temperature > 0, temperature=temperature)
        return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
