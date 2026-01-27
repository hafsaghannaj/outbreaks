from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def build_system_prompt(knowledge_text: str) -> str:
    return (
        "You are the Outbreaks CAG assistant. Use only the provided context to answer. "
        "If the answer is not present, say you don't know based on the provided context.\n\n"
        f"Context:\n{knowledge_text.strip()}\n\n"
        "Question:"
    )


def get_kv_cache(model, tokenizer, prompt: str) -> DynamicCache:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    cache = outputs.past_key_values
    if not isinstance(cache, DynamicCache):
        cache = DynamicCache.from_legacy_cache(cache)
    return cache


def _slice_cache_tensor(tensor: Tensor, origin_len: int) -> Tensor:
    if tensor.dim() == 4:
        if tensor.shape[-2] >= origin_len:
            return tensor[..., :origin_len, :].contiguous()
        if tensor.shape[-1] >= origin_len:
            return tensor[..., :origin_len].contiguous()
    if tensor.dim() == 3:
        if tensor.shape[-2] >= origin_len:
            return tensor[:, :origin_len, :].contiguous()
        if tensor.shape[-1] >= origin_len:
            return tensor[:, :, :origin_len].contiguous()
    return tensor


def clean_up(cache: DynamicCache, origin_len: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(origin_len)
        return
    if hasattr(cache, "truncate"):
        cache.truncate(origin_len)
        return
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for idx in range(len(cache.key_cache)):
            cache.key_cache[idx] = _slice_cache_tensor(cache.key_cache[idx], origin_len)
            cache.value_cache[idx] = _slice_cache_tensor(cache.value_cache[idx], origin_len)
        return
    raise RuntimeError("DynamicCache cleanup failed: unsupported cache structure.")


def generate_greedy(
    model,
    input_ids: Tensor,
    past_key_values: DynamicCache,
    max_new_tokens: int,
) -> Tensor:
    generated = []
    cur_input = input_ids
    cache = past_key_values
    eos_token_id = getattr(model.config, "eos_token_id", None)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=cur_input, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            cur_input = next_token
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
    if generated:
        return torch.cat([input_ids] + generated, dim=-1)
    return input_ids


def save_cache(cache: DynamicCache, origin_len: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache": cache, "origin_len": origin_len}, path)


def load_cache(path: Path, device: torch.device) -> Tuple[DynamicCache, int]:
    payload = torch.load(path, map_location=device)
    cache = payload["cache"]
    origin_len = int(payload["origin_len"])
    if hasattr(cache, "to"):
        cache = cache.to(device)
    return cache, origin_len


class OutbreaksCAG:
    def __init__(
        self,
        model_name: Optional[str] = None,
        knowledge_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_name = model_name or os.getenv("CAG_MODEL_NAME", "gpt2")
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        repo_root = Path(__file__).resolve().parents[3]
        self.knowledge_dir = knowledge_dir or (repo_root / "knowledge")
        self.cache_dir = cache_dir or (Path(os.getenv("CAG_CACHE_DIR")) if os.getenv("CAG_CACHE_DIR") else None)

        self.model = None
        self.tokenizer = None
        self.base_cache: Optional[DynamicCache] = None
        self.base_origin_len: int = 0
        self.region_caches: Dict[str, Tuple[DynamicCache, int]] = {}
        self.last_used_region: Optional[str] = None
        self.last_cache_type: str = "base"

    def load_model(self) -> None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Set it to a valid Hugging Face token before loading the CAG model."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            token=hf_token,
        )
        self.model.to(self.device)
        self.model.eval()

    def _ensure_model(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load_model()

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def build_base_cache(self, knowledge_text: str) -> None:
        self._ensure_model()
        prompt = build_system_prompt(knowledge_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        if self.cache_dir:
            cache_path = self.cache_dir / "base_cache.pt"
            if cache_path.exists():
                self.base_cache, self.base_origin_len = load_cache(cache_path, self.device)
                return
        cache = get_kv_cache(self.model, self.tokenizer, prompt)
        self.base_cache = cache
        self.base_origin_len = int(input_ids.shape[-1])
        if self.cache_dir:
            save_cache(cache, self.base_origin_len, cache_path)

    def build_region_cache(self, region_key: str, knowledge_text: str) -> None:
        self._ensure_model()
        prompt = build_system_prompt(knowledge_text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"region_{region_key}.pt"
            if cache_path.exists():
                cache, origin_len = load_cache(cache_path, self.device)
                self.region_caches[region_key] = (cache, origin_len)
                return
        cache = get_kv_cache(self.model, self.tokenizer, prompt)
        origin_len = int(input_ids.shape[-1])
        self.region_caches[region_key] = (cache, origin_len)
        if cache_path:
            save_cache(cache, origin_len, cache_path)

    def _build_question_prompt(self, question: str) -> str:
        return f" {question.strip()}\nAnswer:"

    def ask(self, question: str, region_key: Optional[str] = None) -> str:
        self._ensure_model()
        if self.base_cache is None:
            raise RuntimeError("Base cache not initialized. Call build_base_cache first.")

        cache = self.base_cache
        origin_len = self.base_origin_len
        cache_type = "base"
        used_region = None

        if region_key:
            region_path = self.knowledge_dir / "regions" / f"{region_key}.md"
            if region_path.exists():
                if region_key not in self.region_caches:
                    base_path = self.knowledge_dir / "playbooks" / "general.md"
                    region_text = self._read_text(region_path)
                    base_text = self._read_text(base_path)
                    combined_text = f"{base_text}\n\n{region_text}"
                    self.build_region_cache(region_key, combined_text)
                cache, origin_len = self.region_caches[region_key]
                cache_type = "region"
                used_region = region_key

        clean_up(cache, origin_len)
        question_prompt = self._build_question_prompt(question)
        inputs = self.tokenizer(question_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        full_ids = generate_greedy(self.model, input_ids, cache, self.max_new_tokens)
        answer_ids = full_ids[:, input_ids.shape[-1]:]
        answer = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()

        self.last_used_region = used_region
        self.last_cache_type = cache_type
        return answer
