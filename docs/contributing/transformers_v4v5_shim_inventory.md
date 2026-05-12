# Transformers v4/v5 Compatibility Shim Inventory

<!-- markdownlint-disable MD013 MD024 -->

> **Ticket:** COG-363 — Phase 1: Inventory all Transformers v4/v5 compatibility shims
>
> Every entry below documents a location in the vLLM codebase that contains
> version-branching logic between Transformers v4 and v5. For each shim the
> table records the **file**, **line range**, **purpose**, and what the
> **v5-only codepath** should look like once v4 support is dropped.

---

## 1 — `vllm/transformers_utils/config.py`

### 1a — `ALLOWED_ATTENTION_LAYER_TYPES` import (lines 51-58)

| Field | Value |
| --- | --- |
| **Pattern** | `try/except ImportError` |
| **Purpose** | In v5 the constant was renamed from `ALLOWED_LAYER_TYPES` to `ALLOWED_ATTENTION_LAYER_TYPES`. The shim imports whichever is available. |
| **v5-only** | Keep only `from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES`. Remove the `try/except`. |

### 1b — Deprecation warning (lines 69-74)

| Field | Value |
| --- | --- |
| **Pattern** | `Version(version("transformers")) < Version("5.0.0")` |
| **Purpose** | Emits a deprecation warning when v4 is installed. |
| **v5-only** | Remove the entire `if` block and the warning. |

### 1c — `patch_rope_parameters` (lines 467-500)

| Field | Value |
| --- | --- |
| **Pattern** | `Version(version("transformers")) < Version("5.0.0")` |
| **Purpose** | Two branches: v4 manually patches legacy rope fields into `rope_parameters` dict and calls `patch_legacy_rope_type`; v5 patches non-standard names then calls `config.standardize_rope_params()` / `config.validate_rope()`. |
| **v5-only** | Remove the entire v4 branch (lines 467-489). Keep the v5 `elif` as the sole codepath, converting it to an unconditional `if`. |

---

## 2 — `vllm/model_executor/models/transformers/base.py`

### 2a — `sub_configs` dtype propagation (lines 221-225)

| Field | Value |
| --- | --- |
| **Pattern** | `getattr(self.config, "sub_configs", {})` |
| **Purpose** | In v4 `sub_configs` does not exist; uses `getattr` fallback. Propagates dtype to sub-configs. |
| **v5-only** | Use `self.config.sub_configs` directly (it is always present in v5). Remove `getattr` guard. |

### 2b — `WeightRenaming` vs `.gamma`/`.beta` fallback (lines 324-354)

| Field | Value |
| --- | --- |
| **Pattern** | `Version(transformers.__version__) >= Version("5.0.0")` |
| **Purpose** | v5 uses the new `WeightRenaming` / `get_model_conversion_mapping` API. v4 falls back to replacing `.gamma`→`.weight`, `.beta`→`.bias` suffixes and reading `_checkpoint_conversion_mapping`. |
| **v5-only** | Remove the `else` branch (lines 340-354). Keep only the v5 `WeightRenaming` path. Remove the `if` guard. |

### 2c — `_get_tie_word_embeddings` (lines 387-394)

| Field | Value |
| --- | --- |
| **Pattern** | `getattr(self.text_config, ...)` and `getattr(self.config, ...)` |
| **Purpose** | v4 stores `tie_word_embeddings` on `text_config`; v5 stores it on the top-level `config`. Checks both. |
| **v5-only** | Only check `self.config.tie_word_embeddings`. |

### 2d — Encoder model version gate (lines 554-558)

| Field | Value |
| --- | --- |
| **Pattern** | `self.check_version("5.0.0", "encoder models support")` |
| **Purpose** | Raises `ImportError` if v4 is installed when loading an encoder model. |
| **v5-only** | Remove the `check_version` call entirely — v5 is always available. |

### 2e — `__module__` preservation (lines 270-281)

| Field | Value |
| --- | --- |
| **Pattern** | Comment-only reference |
| **Purpose** | Preserves `__module__` so v5 source-file checks work. Benign in v5-only world but the TODO comment can be cleaned up. |
| **v5-only** | Keep the code; remove the v5-referencing comment if desired. |

### 2f — `check_version` helper (lines 700-708)

| Field | Value |
| --- | --- |
| **Pattern** | `Version(transformers.__version__)` comparison |
| **Purpose** | Static utility that raises `ImportError` if the installed version is below a minimum. Used by MoE, encoder, and multimodal compilation gates. |
| **v5-only** | All callers can drop the `check_version` calls; the helper itself can be removed or kept as a general-purpose guard for future minimum-version bumps. |

---

## 3 — `vllm/transformers_utils/processor.py`

### 3a — `_transformers_v4_compatibility_import` (lines 39-50)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(processing_utils, "ChatTemplateLoadKwargs")` |
| **Purpose** | v4 has `ChatTemplateLoadKwargs`; v5 merged it into `ProcessorChatTemplateKwargs`. Adds an alias if missing so remote-code processors that import the old name still work. |
| **v5-only** | Remove the function and its call on line 84. |

### 3b — `_transformers_v4_compatibility_init` (lines 53-85)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(ProcessorMixin, "optional_attributes")` |
| **Purpose** | v4 `ProcessorMixin.__init__` accepts arbitrary kwargs via `optional_attributes`; v5 removed this. Patches `__init__` to intercept those kwargs. |
| **v5-only** | Remove the function and its call on line 85. (Note: can only be removed once Molmo2 is upstreamed.) |

---

## 4 — `vllm/tokenizers/mistral.py`

### 4a — `MistralCommonBackend` vs `MistralCommonTokenizer` (lines 40-47)

| Field | Value |
| --- | --- |
| **Pattern** | `try/except ImportError` |
| **Purpose** | v5 renamed the class from `MistralCommonTokenizer` to `MistralCommonBackend`. |
| **v5-only** | Keep only `from transformers.tokenization_mistral_common import MistralCommonBackend`. Remove the `try/except`. |

---

## 5 — `vllm/model_executor/model_loader/weight_utils.py`

### 5a — `HF_XET_HIGH_PERFORMANCE` vs `hf_transfer` (lines 80-103)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(huggingface_hub.constants, "HF_XET_HIGH_PERFORMANCE")` |
| **Purpose** | v5 (with newer `huggingface_hub`) supports `HF_XET_HIGH_PERFORMANCE`; v4 falls back to `hf_transfer`. |
| **v5-only** | Remove `enable_hf_transfer()`, the `hasattr` guard, and the `else` branch. Keep only `enable_xet_high_performance()` called unconditionally. |

---

## 6 — `vllm/renderers/hf.py`

### 6a — `return_dict` default in `apply_chat_template` (lines 658-664)

| Field | Value |
| --- | --- |
| **Pattern** | Comment + conditional `return_dict=False` injection |
| **Purpose** | v5 changed the default of `return_dict` to `True`; this forces `False` for consistency. |
| **v5-only** | Keep the `return_dict=False` injection — it is still needed in v5 to get `list[int]` output. Update the comment to remove v4 references. |

---

## 7 — `vllm/config/vllm.py`

### 7a — `tie_word_embeddings` propagation (lines 606-638)

| Field | Value |
| --- | --- |
| **Pattern** | `Version(version("transformers")) >= Version("5.0.0")` |
| **Purpose** | In v5, `tie_word_embeddings` lives on the outer config for multimodal models. This copies it down to the language model sub-config. |
| **v5-only** | Remove the version guard. Keep the propagation logic unconditionally (it is only relevant for v5 behavior). |

---

## 8 — `vllm/transformers_utils/configs/deepseek_vl2.py`

### 8a — `DeepseekVLV2TextConfig` branched definition (lines 90-99)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(DeepseekV2Config, "validate")` |
| **Purpose** | v5 configs are dataclasses with `@strict` validation; `DeepseekVLV2TextConfig` needs the `@strict` decorator and an extra field. v4 aliases it directly. |
| **v5-only** | Keep only the `@strict` class definition. Remove the `hasattr` guard and `else` branch. |

---

## 9 — `vllm/transformers_utils/configs/qwen3_5.py`

### 9a — `validate_layer_type` vs `layer_type_validation` (lines 97-108)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | v5 added `validate_layer_type()` as an instance method on config; v4 uses a standalone `layer_type_validation()` function. |
| **v5-only** | Keep only `self.validate_layer_type()`. Remove the `hasattr` guard and `else` branch. |

### 9b — `super().__init__()` ordering (lines 117-120)

| Field | Value |
| --- | --- |
| **Pattern** | Comment: "Set these AFTER `super().__init__()` because transformers v4's ..." |
| **Purpose** | v4 `PretrainedConfig.__init__` has explicit params with different defaults that overwrite values. Attributes are set after `super().__init__()`. |
| **v5-only** | Verify v5 no longer has this issue; if so, the workaround can be removed or simplified. Update comment. |

---

## 10 — `vllm/model_executor/models/gemma3.py`

### 10a — Rope parameters nested vs flat (lines 162-174)

| Field | Value |
| --- | --- |
| **Pattern** | `if layer_type in config.rope_parameters` |
| **Purpose** | v5 stores rope_parameters as `dict[layer_type, RopeParams]`; v4 stores them as a flat dict. |
| **v5-only** | Keep only the nested lookup `config.rope_parameters[layer_type]`. Remove the flat-dict fallback. |

---

## 11 — `vllm/model_executor/models/ultravox.py`

### 11a — `layer_head_mask` backward compat (lines 400-404, 507-511)

| Field | Value |
| --- | --- |
| **Pattern** | `inspect.signature` check for `layer_head_mask` parameter |
| **Purpose** | v4 `WhisperEncoderLayer.forward` required `layer_head_mask`; v5 removed it. Dynamically checks the signature. |
| **v5-only** | Remove the `inspect.signature` check, the `kwargs` dict, and the `**kwargs` spread. |

---

## 12 — `vllm/model_executor/models/modernbert.py`

### 12a — `layer_types` vs `global_attn_every_n_layers` (lines 87-104)

| Field | Value |
| --- | --- |
| **Pattern** | `getattr(config, "layer_types", None)` |
| **Purpose** | v5 exposes `layer_types` and per-layer-type `rope_parameters`; v4 uses `global_attn_every_n_layers` and separate theta values. |
| **v5-only** | Keep only the `layer_types` branch. Remove the `getattr` guard and `else` fallback. |

---

## 13 — `vllm/model_executor/models/plamo3.py`

### 13a — Rope parameters nested vs flat (lines 161-170)

| Field | Value |
| --- | --- |
| **Pattern** | `if layer_type in config.rope_parameters` |
| **Purpose** | Same pattern as gemma3 — v5 nested, v4 flat. |
| **v5-only** | Keep only the nested lookup. Remove the flat-dict fallback. |

---

## 14 — `vllm/model_executor/models/gemma3n.py`

### 14a — Rope parameters nested vs flat (lines 338-347)

| Field | Value |
| --- | --- |
| **Pattern** | `if layer_type in config.rope_parameters` |
| **Purpose** | Same pattern as gemma3 — v5 nested, v4 flat. |
| **v5-only** | Keep only the nested lookup. Remove the flat-dict fallback. |

---

## 15 — `vllm/model_executor/models/gemma3n_mm.py`

### 15a — Audio tower output format (lines 620-626)

| Field | Value |
| --- | --- |
| **Pattern** | `isinstance(audio_outputs, tuple)` |
| **Purpose** | v4 audio tower returns a tuple `(encodings, mask)`; v5 returns a named output with `.last_hidden_state` / `.audio_mel_mask`. |
| **v5-only** | Keep only the named-output path. Remove the `isinstance` tuple check. |

---

## 16 — `vllm/transformers_utils/processors/pixtral.py`

### 16a — `init_kwargs` back-compat (lines 60-62)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(self.tokenizer, "init_kwargs")` |
| **Purpose** | v4 tokenizers may not have `init_kwargs`; sets it to `{}` if missing. |
| **v5-only** | Remove the `hasattr` guard. `init_kwargs` is always present in v5. |

---

## 17 — `vllm/transformers_utils/processors/voxtral.py`

### 17a — `init_kwargs` back-compat (lines 67-69)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(self.tokenizer, "init_kwargs")` |
| **Purpose** | Same as pixtral — v4 tokenizers may lack `init_kwargs`. |
| **v5-only** | Remove the `hasattr` guard. |

---

## 18 — `vllm/transformers_utils/configs/olmo_hybrid.py`

### 18a — `validate_layer_type` vs `layer_type_validation` (lines 231-239)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | Same as qwen3_5 — v5 method vs v4 standalone function. |
| **v5-only** | Keep only `self.validate_layer_type()`. Remove the `else` branch. |

---

## 19 — `vllm/transformers_utils/configs/qwen3_next.py`

### 19a — `validate_layer_type` vs `layer_type_validation` (lines 255-262)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | Same pattern — v5 instance method vs v4 standalone function. |
| **v5-only** | Keep only `self.validate_layer_type()`. Remove the `else` branch. |

---

## 20 — `vllm/transformers_utils/configs/qwen3_5_moe.py`

### 20a — `validate_layer_type` vs `layer_type_validation` (lines 103-114)

| Field | Value |
| --- | --- |
| **Pattern** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | Same pattern — v5 instance method vs v4 standalone function. |
| **v5-only** | Keep only `self.validate_layer_type()`. Remove the `else` branch. |

### 20b — `super().__init__()` ordering (lines 129-132)

| Field | Value |
| --- | --- |
| **Pattern** | Comment: "Set these AFTER `super().__init__()` because transformers v4's ..." |
| **Purpose** | Same as qwen3_5 — v4 defaults override values in `super().__init__()`. |
| **v5-only** | Verify and simplify/remove the workaround. |

---

## 21 — `vllm/transformers_utils/configs/speculators/base.py`

### 21a — `is_dataclass(PretrainedConfig)` branching (lines 18-25)

| Field | Value |
| --- | --- |
| **Pattern** | `is_dataclass(PretrainedConfig)` |
| **Purpose** | v5 makes `PretrainedConfig` a dataclass with validation; v4 does not. The init is branched accordingly. |
| **v5-only** | Keep only the v5 dataclass path. Remove the `if not is_dataclass` early return. |

---

## 22 — `vllm/transformers_utils/configs/hyperclovax.py`

### 22a — `HCXVisionConfig` vendored config (lines 281-288)

| Field | Value |
| --- | --- |
| **Pattern** | Entire class with docstring referencing v5 `@strict` fix |
| **Purpose** | Vendored config to handle empty init that breaks v5 `@strict` validation. |
| **v5-only** | Remove once HyperCLOVAX is upstreamed (tracking PR: huggingface/transformers#44956). |

---

## 23 — `vllm/model_executor/models/transformers/utils.py`

### 23a — Parallel style name mapping (lines 129-139)

| Field | Value |
| --- | --- |
| **Pattern** | Dict with both v5 and v4 style names |
| **Purpose** | v5 renamed `colwise_rep`→`colwise_gather_output` and `rowwise_rep`→`rowwise_split_input`. Both are kept for backward compat. |
| **v5-only** | Remove `colwise_rep` and `rowwise_rep` entries. Keep only the v5 names. |

---

## 24 — `vllm/model_executor/models/transformers/moe.py`

### 24a — `check_version("5.0.0")` gate (line 126)

| Field | Value |
| --- | --- |
| **Pattern** | `self.check_version("5.0.0", "MoE models support")` |
| **Purpose** | Prevents MoE model loading on v4. |
| **v5-only** | Remove the call. |

### 24b — Fused experts comment (lines 164-166)

| Field | Value |
| --- | --- |
| **Pattern** | Comment referencing "Before Transformers v5" |
| **Purpose** | Documents that fused-expert checkpoints exist for both v4 and v5. |
| **v5-only** | Update comment to remove v4 reference. |

---

## 25 — `vllm/model_executor/models/transformers/multimodal.py`

### 25a — `check_version("5.0.0")` gate (line 311)

| Field | Value |
| --- | --- |
| **Pattern** | `self.check_version("5.0.0", "multimodal encoder compilation support")` |
| **Purpose** | Prevents MM encoder compilation on v4. |
| **v5-only** | Remove the call. |

---

## 26 — `vllm/model_executor/model_loader/gguf_loader.py`

### 26a — `_checkpoint_conversion_mapping` fallback (lines 228-240)

| Field | Value |
| --- | --- |
| **Pattern** | `getattr(dummy_model, "_checkpoint_conversion_mapping", None)` |
| **Purpose** | Uses v4's `_checkpoint_conversion_mapping` to revert HF weight renames in GGUF loading. |
| **v5-only** | Migrate to v5's `WeightRenaming` / `get_model_conversion_mapping` API, or confirm this is still needed for GGUF. |

### 26b — `model.` prefix stripping for v5 multimodal (lines 268-274)

| Field | Value |
| --- | --- |
| **Pattern** | Comment: "In transformers v5, multimodal models wrap all sub-models under `model.`" |
| **Purpose** | Strips outer `model.` prefix from state_dict keys so GGUF tensor names match. |
| **v5-only** | Keep this logic — it is the v5 behavior. Remove the explanatory v5 comment context. |

---

## 27 — `vllm/config/model.py`

### 27a — `rope_parameters` v5 TypedDict handling (lines 2117-2121)

| Field | Value |
| --- | --- |
| **Pattern** | Comment: "In Transformers v5 rope_parameters could be TypedDict or dict[str, TypedDict]" |
| **Purpose** | Normalizes `rope_parameters` to `dict[str, TypedDict]` for uniform verification. |
| **v5-only** | Keep the normalization. Update comment to remove "In Transformers v5" framing. |

---

## 28 — `vllm/model_executor/models/parakeet.py`

### 28a — `convolution_bias` propagation (lines 116-123)

| Field | Value |
| --- | --- |
| **Pattern** | Comment: "In transformers v5 (not v4), `convolution_bias=False` is propagated" |
| **Purpose** | v5 propagates `convolution_bias=False` from Parakeet config to conv1d, which skips registering bias params. The method allows missing bias weights. |
| **v5-only** | Keep the logic (it handles v5 behavior). Update comment. |

---

## 29 — `vllm/model_executor/models/kimi_k25.py`

### 29a — Token ID remapping (lines 116-123)

| Field | Value |
| --- | --- |
| **Pattern** | Comment: "transformers v5 may remap token IDs vs config.json" |
| **Purpose** | Resolves token IDs from the tokenizer because v5 may remap them differently than config.json. |
| **v5-only** | Keep the resolution logic (it is the correct v5 approach). Update comment. |

---

## Summary Statistics

| Category | Count |
| --- | --- |
| Total shim locations | 35 |
| `try/except ImportError` | 2 |
| `Version(...)` comparison | 5 |
| `hasattr(...)` feature detection | 8 |
| `isinstance(...)` output format check | 1 |
| `inspect.signature(...)` check | 2 |
| `is_dataclass(...)` check | 1 |
| `check_version(...)` gate | 3 |
| Comment-only / style mapping | 6 |
| v4 workaround (init ordering, etc.) | 2 |
| Vendored config for v5 compat | 1 |
| Unconditional v5 compat (keep as-is) | 4 |

### Shims safe to remove outright (v4-only code)

1. `config.py` — `ALLOWED_LAYER_TYPES` import fallback
2. `config.py` — deprecation warning
3. `config.py` — v4 rope patching branch
4. `base.py` — `.gamma`/`.beta` fallback + `_checkpoint_conversion_mapping`
5. `base.py` — `check_version` calls (3 locations)
6. `processor.py` — both `_transformers_v4_compatibility_*` functions
7. `mistral.py` — `MistralCommonTokenizer` fallback
8. `weight_utils.py` — `hf_transfer` fallback
9. `deepseek_vl2.py` — v4 alias branch
10. `qwen3_5.py`, `qwen3_next.py`, `qwen3_5_moe.py`, `olmo_hybrid.py` — `layer_type_validation` fallbacks (4 locations)
11. `speculators/base.py` — v4 init branch
12. `gemma3.py`, `gemma3n.py`, `plamo3.py` — flat rope_parameters fallbacks (3 locations)
13. `modernbert.py` — `global_attn_every_n_layers` fallback
14. `ultravox.py` — `layer_head_mask` compat (2 locations)
15. `pixtral.py`, `voxtral.py` — `init_kwargs` guards (2 locations)
16. `gemma3n_mm.py` — tuple output check
17. `utils.py` — `colwise_rep`/`rowwise_rep` style names
18. `moe.py` — `check_version` gate
19. `multimodal.py` — `check_version` gate

### Shims to keep (v5 behavior, just update comments)

1. `hf.py` — `return_dict=False` injection (needed in v5)
2. `vllm.py` — `tie_word_embeddings` propagation (v5 behavior)
3. `gguf_loader.py` — `model.` prefix stripping (v5 behavior)
4. `model.py` — `rope_parameters` normalization (v5 behavior)
5. `parakeet.py` — `convolution_bias` handling (v5 behavior)
6. `kimi_k25.py` — token ID resolution (v5 behavior)

### Shims blocked on upstream

1. `processor.py` 3b — blocked on Molmo2 upstreaming
2. `hyperclovax.py` — blocked on huggingface/transformers#44956
