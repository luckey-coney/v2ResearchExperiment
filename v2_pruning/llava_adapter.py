"""Offline single-image adapter for original LLaVA-1.5 Llama checkpoints.

Reuses the pinned AgilePruner model, conversation templates, image preprocessing,
and tokenization. Loading/feature extraction follow its Apache-2.0 builder.py,
clip_encoder.py and model_vqa_loader.py. Model generation uses Transformers'
GenerationMixin, bypassing AgilePruner's fixed-budget generate wrapper.
See THIRD_PARTY_NOTICES.md for sources and modifications.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass

from .timing import _StageTimer
from .pairing import verify_pairing


def enforce_offline() -> None:
    """Must run before importing Transformers or the vendored model package."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def preflight_runtime() -> tuple:
    """Import required lazy backends before claiming output; do not load weights."""
    enforce_offline()
    installed = importlib.metadata.version("transformers")
    if installed != "4.37.2":
        raise RuntimeError(f"This adapter targets transformers==4.37.2, found {installed}")
    # Transformers' lazy imports/low_cpu_mem_usage can otherwise defer these
    # missing or broken native dependencies until checkpoint loading.
    import torch
    import torchvision
    import accelerate
    import sentencepiece
    import tokenizers
    import safetensors.torch
    from huggingface_hub.utils.sha import sha_fileobj
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
    from transformers.utils import is_accelerate_available
    if not is_accelerate_available():
        raise RuntimeError("A compatible accelerate backend is required for low_cpu_mem_usage")
    vendor = Path(__file__).resolve().parents[1] / "third_party" / "agilepruner"
    existing = sys.modules.get("llava")
    if existing is not None:
        existing_path = Path(existing.__file__).resolve()
        if vendor.resolve() not in existing_path.parents:
            raise RuntimeError("Another llava package is already imported; use a fresh process")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    return torch, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, LlavaConfig, LlavaLlamaForCausalLM


def validate_loading_info(info: dict, component: str, allowed_unexpected: tuple[str, ...] = ()) -> dict:
    """Do not let from_pretrained's random missing-weight initialization look valid."""
    buffer_suffixes = (".rotary_emb.inv_freq", ".rotary_emb.cos_cached", ".rotary_emb.sin_cached")
    missing = [key for key in info.get("missing_keys", []) if not key.endswith(buffer_suffixes)]
    unexpected = [key for key in info.get("unexpected_keys", [])
                  if not key.endswith(buffer_suffixes) and not key.startswith(allowed_unexpected)]
    mismatched = info.get("mismatched_keys", [])
    errors = info.get("error_msgs", [])
    if missing or unexpected or mismatched or errors:
        raise ValueError(f"Incomplete/incompatible {component} checkpoint: missing={missing[:8]}, "
                         f"unexpected={unexpected[:8]}, mismatched={mismatched[:4]}, errors={errors}")
    return {"missing_keys": info.get("missing_keys", []),
            "allowed_unexpected_keys": info.get("unexpected_keys", []),
            "mismatched_keys": mismatched, "error_msgs": errors}


@dataclass
class PreparedInput:
    v0: object
    inputs_embeds: object
    token_ids: object
    position_ids: object
    visual_positions: list[int]
    question_positions: list[int]
    grid_shape: tuple[int, int] | None
    pixel_values: object
    original_size: tuple[int, int]
    prompt: str
    grid_error: str | None = None
    stage_timings_ms: dict | None = None


def _common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def question_token_span(full: list[int], before: list[int], through: list[int],
                        question: str | None = None) -> range:
    """Align a character-delimited question with slow-tokenizer BPE boundaries.

    A token crossing either character boundary belongs to the question span.
    Prefix comparisons avoid assuming that separately tokenized pieces concatenate.
    """
    # Token boundaries can merge whitespace with template markers. An empty or
    # whitespace-only instruction has no valid query even if such a token exists.
    if question is not None and not question.strip():
        return range(0)
    start = _common_prefix(full, before)
    shared_end = _common_prefix(full, through)
    end = shared_end + int(shared_end < len(through))
    if not 0 <= start < end <= len(full):
        return range(0)
    return range(start, end)


def resolve_grid_shape(height: int, width: int, patch_size: int, visual_count: int
                       ) -> tuple[tuple[int, int] | None, str | None]:
    """Return a trusted spatial grid or a reason to keep the full visual input.

    This only validates the spatial mapping; it does not guarantee that the
    encoder itself can process arbitrary image dimensions.
    """
    values = (height, width, patch_size, visual_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        return None, "invalid image dimensions, patch size, or visual token count"
    if height % patch_size or width % patch_size:
        return None, "processed image size is incompatible with the CLIP patch size"
    shape = (height // patch_size, width // patch_size)
    if shape[0] * shape[1] != visual_count:
        return None, "visual rows and the actual spatial grid disagree"
    return shape, None


class LlavaAdapter:
    """Shares the loaded model's parameters; no second language model is created."""

    def __init__(self, model_path: Path, vision_path: Path, device: str = "cuda:0", *,
                 pairing_profile: str = "auto"):
        enforce_offline()
        (torch, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel,
         LlavaConfig, LlavaLlamaForCausalLM) = preflight_runtime()

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("This first adapter requires a CUDA device and FP16 model inference")
        self.torch = torch
        self.device = torch.device(device)
        # Verify publisher-pinned bytes before replacing the config's remote ID
        # with an offline path or loading any model/Projector/CLIP weights.
        self.pairing_info = verify_pairing(model_path, vision_path, pairing_profile)
        config = LlavaConfig.from_pretrained(str(model_path), local_files_only=True)
        if config.mm_vision_tower != self.pairing_info["original_mm_vision_tower"]:
            raise ValueError("Original vision identity changed after pairing verification")
        if not hasattr(config, "mm_hidden_size") or not hasattr(config, "mm_vision_tower"):
            raise ValueError("Use original LLaVA-1.5 weights, not a llava-hf converted checkpoint")
        if getattr(config, "s2", False):
            raise ValueError("Multi-scale S2 vision is outside this single-grid adapter")
        if getattr(config, "mm_vision_select_feature", "patch") != "patch":
            raise ValueError("The adapter requires patch-only CLIP features")
        if getattr(config, "image_aspect_ratio", "square") not in {"pad", "square"}:
            raise ValueError("Only a single square/padded image grid is supported")
        config.mm_vision_tower = str(vision_path.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), use_fast=False, local_files_only=True
        )
        self.model, loading = LlavaLlamaForCausalLM.from_pretrained(
            str(model_path), config=config, local_files_only=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            attn_implementation="eager", visual_token_num=576, output_loading_info=True,
            use_safetensors=False,
        )
        self.loading_info = {"language_projector": validate_loading_info(
            loading, "language/projector", ("model.vision_tower.",)
        )}
        self.model.to(self.device)
        tower = self.model.get_vision_tower()
        # Use the same native CLIP classes as the reused tower, collecting loading
        # information that the upstream load_model wrapper would otherwise discard.
        tower.image_processor = CLIPImageProcessor.from_pretrained(str(vision_path), local_files_only=True)
        tower.vision_tower, loading = CLIPVisionModel.from_pretrained(
            str(vision_path), local_files_only=True, output_loading_info=True, torch_dtype=torch.float16,
            use_safetensors=False,
        )
        self.loading_info["vision"] = validate_loading_info(
            loading, "CLIP vision", ("text_model.", "text_projection.", "visual_projection.", "logit_scale")
        )
        tower.vision_tower.requires_grad_(False)
        tower.is_loaded = True
        tower.to(device=self.device, dtype=torch.float16)
        self.image_processor = tower.image_processor
        self.model.eval().requires_grad_(False)
        self.context_length = int(config.max_position_embeddings)
        self._check_special_tokens()

    def _check_special_tokens(self) -> None:
        from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        tokens = []
        if getattr(self.model.config, "mm_use_im_patch_token", True):
            tokens.append(DEFAULT_IMAGE_PATCH_TOKEN)
        if getattr(self.model.config, "mm_use_im_start_end", False):
            tokens.extend((DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN))
        self.tokenizer.add_tokens(tokens, special_tokens=True)
        if len(self.tokenizer) > self.model.get_input_embeddings().num_embeddings:
            raise ValueError("Checkpoint/tokenizer mismatch; refusing to initialize new token weights")

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def prepare(self, image, question: str, conv_mode: str = "llava_v1") -> PreparedInput:
        """Reuse upstream preprocessing; encode/project once, before any selection."""
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        stage_timings = {}
        with _StageTimer(stage_timings, "text_input", self.device):
            if not isinstance(question, str):
                raise ValueError("question must be a string; empty text uses the full-keep fallback")
            if DEFAULT_IMAGE_TOKEN in question:
                raise ValueError("Question must not contain an additional <image> placeholder")
            if conv_mode not in conv_templates:
                raise ValueError(f"Unknown upstream conversation template: {conv_mode}")
            marker = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, "mm_use_im_start_end", False):
                marker = DEFAULT_IM_START_TOKEN + marker + DEFAULT_IM_END_TOKEN
            user_message = marker + "\n" + question
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], user_message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            user_start = prompt.rfind(user_message)
            if user_start < 0:
                raise ValueError("Cannot locate the user message in the native template")
            q_start = user_start + len(marker) + 1
            encode = lambda value: tokenizer_image_token(value, self.tokenizer, IMAGE_TOKEN_INDEX)
            ids = encode(prompt)
            q_tokens = question_token_span(ids, encode(prompt[:q_start]),
                                           encode(prompt[:q_start + len(question)]), question)
            image_slots = [i for i, token in enumerate(ids) if token == IMAGE_TOKEN_INDEX]
            if len(image_slots) != 1:
                raise ValueError("Exactly one image placeholder is required")
            image_slot = image_slots[0]

        with _StageTimer(stage_timings, "preprocessing", self.device):
            pixels = process_images([image], self.image_processor, self.model.config)
            if not torch.is_tensor(pixels) or pixels.ndim != 4 or pixels.shape[0] != 1:
                raise ValueError("Preprocessor did not return one image grid")
            pixels = pixels.to(device=self.device, dtype=torch.float16)
            tower = self.model.get_vision_tower()
            patch = int(tower.config.patch_size)
            h, w = pixels.shape[-2:]
        with torch.inference_mode():
            # Adapted from upstream clip_encoder.feature_select, without its unneeded CLS attention output.
            with _StageTimer(stage_timings, "vision_encoder", self.device):
                vision = tower.vision_tower(pixels, output_hidden_states=True, output_attentions=False)
                features = vision.hidden_states[tower.select_layer][:, 1:]
            with _StageTimer(stage_timings, "projector", self.device):
                v0 = self.model.get_model().mm_projector(features).squeeze(0)
            with _StageTimer(stage_timings, "text_input", self.device):
                n = int(v0.shape[0])
                grid_shape, grid_error = resolve_grid_shape(h, w, patch, n)
                ids_tensor = torch.tensor(ids, device=self.device, dtype=torch.long)
                embed = self.model.get_model().embed_tokens
                full_embeds = torch.cat((embed(ids_tensor[:image_slot]), v0,
                                         embed(ids_tensor[image_slot + 1:])), dim=0).unsqueeze(0)
                safe_id = self.tokenizer.bos_token_id
                if safe_id is None:
                    safe_id = self.tokenizer.eos_token_id
                if safe_id is None:
                    raise ValueError("A valid placeholder token id is required for generation bookkeeping")
                safe_ids = ids[:image_slot] + [safe_id] * n + ids[image_slot + 1:]
                q_positions = [i if i < image_slot else i + n - 1 for i in q_tokens if i != image_slot]
                length = len(safe_ids)
                if length > self.context_length:
                    raise ValueError("The full probe prompt exceeds the model context; refusing to truncate text")
                return PreparedInput(
                    v0=v0, inputs_embeds=full_embeds,
                    token_ids=torch.tensor([safe_ids], device=self.device, dtype=torch.long),
                    position_ids=torch.arange(length, device=self.device).unsqueeze(0),
                    visual_positions=list(range(image_slot, image_slot + n)),
                    question_positions=q_positions, grid_shape=grid_shape, pixel_values=pixels,
                    original_size=image.size, prompt=prompt, grid_error=grid_error,
                    stage_timings_ms=stage_timings,
                )

    def canvas(self, prepared: PreparedInput):
        """Recover the actual model-input pixels for grid overlays, after timing."""
        from PIL import Image
        torch = self.torch
        pixels = prepared.pixel_values[0].float()
        mean = torch.tensor(self.image_processor.image_mean, device=pixels.device)[:, None, None]
        std = torch.tensor(self.image_processor.image_std, device=pixels.device)[:, None, None]
        array = ((pixels * std + mean).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(array, mode="RGB")

    def generate(self, prepared: PreparedInput, selected: list[int], max_new_tokens: int,
                 position_policy: str = "compact") -> dict:
        """Reuse GenerationMixin and the native KV cache with a shortened input.

        Native LLaVA reconstructs compact sequence positions after pruning. Spatial
        coordinates and original sequence indices remain in PreparedInput. The native
        generation helper derives continuation position_ids from the attention mask;
        a fixed prompt position_ids tensor must not be reused on every decode step.
        This adapter is sequential, batch=1, greedy, one beam.
        """
        import torch
        from transformers.generation.utils import GenerationMixin

        if position_policy != "compact":
            raise ValueError("Pinned LLaVA uses compact sequence positions; sparse RoPE positions are unsupported")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        n = len(prepared.visual_positions)
        if (selected != sorted(set(selected)) or
                any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n for i in selected)):
            raise ValueError("Selected indices must be sorted, unique original visual indices")
        self.synchronize()
        started = time.perf_counter()
        keep_visual = {prepared.visual_positions[i] for i in selected}
        visual = set(prepared.visual_positions)
        keep = [i for i in range(prepared.token_ids.shape[1]) if i not in visual or i in keep_visual]
        keep_tensor = torch.tensor(keep, dtype=torch.long, device=self.device)
        inputs_embeds = prepared.inputs_embeds.index_select(1, keep_tensor)
        input_ids = prepared.token_ids.index_select(1, keep_tensor)
        prompt_len = input_ids.shape[1]
        if prompt_len + max_new_tokens > self.context_length:
            raise ValueError("Prompt positions plus requested output exceed the model context")

        class FirstTokenTimer:
            first = None
            prompt_pending = True

            def put(timer, value):
                if timer.prompt_pending:
                    timer.prompt_pending = False
                elif timer.first is None:
                    timer.first = time.perf_counter()

            def end(timer):
                pass

        timer = FirstTokenTimer()
        forward_timings = {"generation_prefill_ms": None, "generation_decode_forward_ms": None}
        active, prefill_seen = None, False

        def before_forward(module, args, kwargs):
            nonlocal active, prefill_seen
            if not prefill_seen and kwargs.get("past_key_values") is not None:
                return
            stage = "generation_decode_forward_ms" if prefill_seen else "generation_prefill_ms"
            prefill_seen = True
            context = _StageTimer(forward_timings, stage, self.device)
            context.__enter__()
            active = context

        def after_forward(module, args, kwargs, output):
            nonlocal active
            if active is not None:
                active.__exit__(None, None, None)
                active = None

        decoder = (self.model.get_model() if callable(getattr(self.model, "get_model", None))
                   else self.model.model)
        hooks = []
        try:
            hooks.append(decoder.register_forward_pre_hook(before_forward, with_kwargs=True))
            hooks.append(decoder.register_forward_hook(after_forward, with_kwargs=True, always_call=True))
            with torch.inference_mode():
                output = GenerationMixin.generate(
                    self.model, inputs=input_ids, inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False, num_beams=1, use_cache=True,
                    repetition_penalty=1.0, no_repeat_ngram_size=0,
                    max_new_tokens=max_new_tokens, streamer=timer,
                    pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                  else self.tokenizer.eos_token_id),
                    return_dict_in_generate=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
        self.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        generated_ids = output.sequences[0, prompt_len:]
        return {
            "answer": self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
            "generated_tokens": int(generated_ids.numel()),
            "generation_ms": elapsed,
            "generation_ttft_ms": None if timer.first is None else (timer.first - started) * 1000,
            **forward_timings,
            "generation_decode_ms": None,  # Sampling, LM head and streamer are not isolated.
            "input_sequence_length": int(prompt_len),
            "position_policy": position_policy,
        }
