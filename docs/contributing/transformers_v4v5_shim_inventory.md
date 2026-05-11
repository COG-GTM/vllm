# Transformers v4/v5 Compatibility Shim Inventory

<!-- markdownlint-disable MD013 MD033 -->

> **Ticket:** COG-358 — Phase 1: Inventory all Transformers v4/v5 compatibility shims
>
> **Purpose:** Comprehensive audit of every file in the vLLM codebase containing
> Transformers v4/v5 version-branching logic. Each entry documents the shim's
> location, mechanism, purpose, and recommended v5-only codepath.

## Legend

| Column | Description |
| --- | --- |
| **File** | Path relative to repo root |
| **Lines** | Approximate line range of the shim |
| **Mechanism** | How the branch is implemented (e.g. `try/except`, `Version()` check, `hasattr`) |
| **Purpose** | What the shim does |
| **v5-only codepath** | What the code should look like once v4 support is dropped |

---

## 1 — `vllm/transformers_utils/` (utilities layer)

### 1.1 `vllm/transformers_utils/config.py`

#### Shim A — `ALLOWED_ATTENTION_LAYER_TYPES` import (lines 51-58)

| Field | Value |
| --- | --- |
| **Mechanism** | `try/except ImportError` |
| **Purpose** | In v5 the constant was renamed from `ALLOWED_LAYER_TYPES` to `ALLOWED_ATTENTION_LAYER_TYPES`. The shim imports the v5 name and falls back to the v4 name with an alias. |
| **v5-only** | Direct import: `from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES`. Remove the `try/except` block. |

#### Shim B — Deprecation warning (lines 69-74)

| Field | Value |
| --- | --- |
| **Mechanism** | `Version(version("transformers")) < Version("5.0.0")` |
| **Purpose** | Logs a warning that v4 support is deprecated and will be removed. |
| **v5-only** | Remove the entire `if` block and the warning. |

#### Shim C — `patch_rope_parameters()` (lines 467-500)

| Field | Value |
| --- | --- |
| **Mechanism** | `Version(version("transformers")) < Version("5.0.0")` |
| **Purpose** | Provides backwards compatibility for RoPE configuration. On v4: patches legacy `rope_scaling` fields into `rope_parameters`, handles nested `rope_parameters` from v5 configs, and calls `patch_legacy_rope_type`. On v5: patches non-standard field names, then calls `config.standardize_rope_params()` and `config.validate_rope()`. |
| **v5-only** | Keep only the `elif` branch (v5 path): patch non-standard names, call `standardize_rope_params()` and `validate_rope()`. Remove all v4 legacy field handling. |

### 1.2 `vllm/transformers_utils/processor.py`

#### Shim A — `_transformers_v4_compatibility_import()` (lines 39-50)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr` check on `transformers.processing_utils` |
| **Purpose** | In v5, `ChatTemplateLoadKwargs` was merged into `ProcessorChatTemplateKwargs` and removed. This shim adds `ChatTemplateLoadKwargs` as an alias so remote-code processors that still import it don't break. |
| **v5-only** | Remove the function and its call at line 84. Remote-code processors that depend on `ChatTemplateLoadKwargs` would need updating. |

#### Shim B — `_transformers_v4_compatibility_init()` (lines 53-85)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(ProcessorMixin, "optional_attributes")` |
| **Purpose** | In v5, `ProcessorMixin.__init__` no longer accepts arbitrary keyword arguments. This patches `__init__` to intercept `optional_attributes` and set them manually before calling the original, for remote-code processors like Molmo2. |
| **v5-only** | Remove the function and its call at line 85. Can be removed once `Molmo2ForConditionalGeneration` is upstreamed to Transformers. |

### 1.3 `vllm/transformers_utils/processors/pixtral.py`

#### Shim — `init_kwargs` fallback (lines 60-62)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr` check |
| **Purpose** | Back-compatibility for v4 tokenizers that lack the `init_kwargs` attribute. Sets it to `{}` if missing. |
| **v5-only** | Remove the `if not hasattr(...)` block. |

### 1.4 `vllm/transformers_utils/processors/voxtral.py`

#### Shim — `init_kwargs` fallback (lines 67-69)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr` check |
| **Purpose** | Identical to `pixtral.py` — back-compatibility for v4 tokenizers missing `init_kwargs`. |
| **v5-only** | Remove the `if not hasattr(...)` block. |

### 1.5 `vllm/transformers_utils/configs/deepseek_vl2.py`

#### Shim — `DeepseekVLV2TextConfig` branched definition (lines 90-99)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(DeepseekV2Config, "validate")` |
| **Purpose** | In v5, `PretrainedConfig` uses `@strict` dataclass validation. The shim defines `DeepseekVLV2TextConfig` with `@strict` and an extra `kv_lora_rank` field on v5, and aliases it to `DeepseekV2Config` on v4. |
| **v5-only** | Keep only the `@strict` class definition. Remove the `else` branch. |

### 1.6 `vllm/transformers_utils/configs/qwen3_5.py`

#### Shim A — `validate_layer_type` vs `layer_type_validation` (lines 97-108)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | In v5, layer type validation moved from a standalone function `layer_type_validation` to a method `validate_layer_type()` on `PretrainedConfig`. Also passes `ignore_keys_at_rope_validation` in v5. |
| **v5-only** | Call `self.validate_layer_type()` directly with `ignore_keys_at_rope_validation`. Remove the `else` branch. |

#### Shim B — Post-`super().__init__()` attribute override (lines 117-123)

| Field | Value |
| --- | --- |
| **Mechanism** | Comment-documented workaround |
| **Purpose** | v4's `PretrainedConfig.__init__` has explicit params like `tie_word_embeddings=True` that would overwrite values set before the call. Attributes are set after `super().__init__()` to preserve intended values. |
| **v5-only** | Verify that v5's dataclass-based `PretrainedConfig` does not have this issue. If safe, move attribute assignment before `super().__init__()`. |

### 1.7 `vllm/transformers_utils/configs/qwen3_5_moe.py`

#### Shim A — `validate_layer_type` vs `layer_type_validation` (lines 103-114)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | Same pattern as `qwen3_5.py` — routes to v5 method or v4 function. |
| **v5-only** | Call `self.validate_layer_type()` directly. Remove the `else` branch. |

#### Shim B — Post-`super().__init__()` attribute override (lines 129-134)

| Field | Value |
| --- | --- |
| **Mechanism** | Comment-documented workaround |
| **Purpose** | Same pattern as `qwen3_5.py` — avoids v4 default overwriting. |
| **v5-only** | Same as `qwen3_5.py` Shim B. |

### 1.8 `vllm/transformers_utils/configs/qwen3_next.py`

#### Shim — `validate_layer_type` vs `layer_type_validation` (lines 255-262)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | Same pattern as `qwen3_5.py`. |
| **v5-only** | Call `self.validate_layer_type()` directly. Remove the `else` branch. |

### 1.9 `vllm/transformers_utils/configs/olmo_hybrid.py`

#### Shim — `validate_layer_type` vs `layer_type_validation` (lines 231-239)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(self, "validate_layer_type")` |
| **Purpose** | Same pattern — v5 method vs v4 function for layer type validation. |
| **v5-only** | Call `self.validate_layer_type()` directly. Remove the `else` branch. |

### 1.10 `vllm/transformers_utils/configs/speculators/base.py`

#### Shim — `SpeculatorsConfig.__init__` (lines 18-33)

| Field | Value |
| --- | --- |
| **Mechanism** | `is_dataclass(PretrainedConfig)` |
| **Purpose** | In v4, `PretrainedConfig.__init__` sets all kwargs as attributes. In v5, it is a dataclass with validation, so unknown kwargs must be set via `setattr` before calling `super().__init__()`. |
| **v5-only** | Keep only the v5 path (the dataclass branch). Remove the `if not is_dataclass(...)` early return. |

### 1.11 `vllm/transformers_utils/configs/hyperclovax.py`

#### Shim — `HCXVisionConfig` vendored config (lines 280-288)

| Field | Value |
| --- | --- |
| **Mechanism** | Vendored class with defensive `__init__` |
| **Purpose** | The original remote-code config does not handle empty initialization (`text_config=None`), which breaks v5's `@strict` validation. This vendored version adds the fix. |
| **v5-only** | Remove once HyperCLOVAX is upstreamed to Transformers (tracking PR: <https://github.com/huggingface/transformers/pull/44956>). |

---

## 2 — `vllm/model_executor/` (models and loaders)

### 2.1 `vllm/model_executor/models/transformers/base.py`

#### Shim A — `sub_configs` dtype propagation (lines 221-225)

| Field | Value |
| --- | --- |
| **Mechanism** | `getattr(self.config, "sub_configs", {})` |
| **Purpose** | Propagates dtype to sub-configs. Marked `TODO(hmellor): Remove this when Transformers v4 support is dropped`. In v5, `sub_configs` is always present on `PretrainedConfig`. |
| **v5-only** | Access `self.config.sub_configs` directly without `getattr` fallback. |

#### Shim B — `__module__` preservation (lines 278-281)

| Field | Value |
| --- | --- |
| **Mechanism** | Explicit `__module__` assignment |
| **Purpose** | v5 added source-file checks (e.g. `_can_set_experts_implementation`) that inspect `__module__`. The wrapper class must preserve the original module for these checks to work. |
| **v5-only** | Keep as-is — this is a v5 requirement, not a v4 compat shim. No cleanup needed. |

#### Shim C — `_create_hf_to_vllm_mapper` weight renaming (lines 324-354)

| Field | Value |
| --- | --- |
| **Mechanism** | `Version(transformers.__version__) >= Version("5.0.0")` |
| **Purpose** | In v5, uses `WeightRenaming` from `transformers.conversion_mapping` for weight name mapping. In v4, uses regex replacement for legacy `.gamma`/`.beta` suffixes and `_checkpoint_conversion_mapping`. |
| **v5-only** | Keep only the v5 branch using `get_model_conversion_mapping` and `WeightRenaming`. Remove the `.gamma`/`.beta` regex and `_checkpoint_conversion_mapping` fallback (both marked with TODO). |

#### Shim D — `_get_tie_word_embeddings` (lines 387-394)

| Field | Value |
| --- | --- |
| **Mechanism** | `getattr` from two locations |
| **Purpose** | v4 stores `tie_word_embeddings` on `text_config`, v5 stores it on `config`. This checks both. |
| **v5-only** | Read only from `self.config.tie_word_embeddings`. Remove the v4 `text_config` fallback. |

#### Shim E — Encoder models version gate (lines 556-558)

| Field | Value |
| --- | --- |
| **Mechanism** | `self.check_version("5.0.0", ...)` |
| **Purpose** | Encoder models in the Transformers backend require v5. Raises `ImportError` on v4. |
| **v5-only** | Remove the `check_version` call — all users will be on v5. |

#### Shim F — `check_version` utility (lines 700-708)

| Field | Value |
| --- | --- |
| **Mechanism** | `Version(transformers.__version__)` comparison |
| **Purpose** | Static utility that gates features behind a minimum Transformers version. Used by multiple shims. |
| **v5-only** | Remove the method entirely once all callers are removed. Or keep for future version gating if needed. |

### 2.2 `vllm/model_executor/models/transformers/utils.py`

#### Shim — Parallel style name aliases (lines 129-138)

| Field | Value |
| --- | --- |
| **Mechanism** | Dict mapping with both v4 and v5 style names |
| **Purpose** | v5 renamed `colwise_rep` → `colwise_gather_output` and `rowwise_rep` → `rowwise_split_input`. Both old and new names are supported. |
| **v5-only** | Remove the v4 entries (`colwise_rep`, `rowwise_rep`). Keep only v5 names. |

### 2.3 `vllm/model_executor/models/transformers/moe.py`

#### Shim A — MoE version gate (line 126)

| Field | Value |
| --- | --- |
| **Mechanism** | `self.check_version("5.0.0", "MoE models support")` |
| **Purpose** | MoE models in the Transformers backend require v5. Raises on v4. |
| **v5-only** | Remove the `check_version` call. |

#### Shim B — Fused experts comment (lines 164-166)

| Field | Value |
| --- | --- |
| **Mechanism** | Comment only |
| **Purpose** | Documents that fused expert checkpoints were released after v5 or re-saved. No runtime branching. |
| **v5-only** | No code change needed. Update comment to remove "Before Transformers v5" reference. |

### 2.4 `vllm/model_executor/models/transformers/multimodal.py`

#### Shim — Multimodal encoder compilation version gate (line 311)

| Field | Value |
| --- | --- |
| **Mechanism** | `self.check_version("5.0.0", ...)` |
| **Purpose** | Multimodal encoder compilation requires v5. |
| **v5-only** | Remove the `check_version` call. |

### 2.5 `vllm/model_executor/models/gemma3.py`

#### Shim — RoPE parameters nested vs flat dict (lines 162-174)

| Field | Value |
| --- | --- |
| **Mechanism** | `layer_type in config.rope_parameters` |
| **Purpose** | In v5, `rope_parameters` is a `dict[str, TypedDict]` keyed by layer type. In v4, it is a flat `TypedDict`. The shim checks if the layer type is a key to determine the format. |
| **v5-only** | Access `config.rope_parameters[layer_type]` directly. Remove the `else` branch with flat-dict handling and local-attention override. |

### 2.6 `vllm/model_executor/models/gemma3n.py`

#### Shim — RoPE parameters nested vs flat dict (lines 340-349)

| Field | Value |
| --- | --- |
| **Mechanism** | `layer_type in config.rope_parameters` |
| **Purpose** | Same pattern as `gemma3.py`. Uses `.copy()` for the flat dict and overrides `rope_theta` for sliding attention. |
| **v5-only** | Access `config.rope_parameters[layer_type]` directly. Remove the flat-dict fallback. |

### 2.7 `vllm/model_executor/models/gemma3n_mm.py`

#### Shim A — Weight prefix mapping (lines 486-488)

| Field | Value |
| --- | --- |
| **Mechanism** | `orig_to_new_prefix` in `WeightsMapper` |
| **Purpose** | Maps checkpoint names saved after Transformers v4.52 (`model.embed_audio.` → `embed_audio.`, etc.). |
| **v5-only** | Keep — these mappings handle checkpoint format changes, not runtime branching. They remain needed for loading older checkpoints. |

#### Shim B — Audio tower output format (lines 619-626)

| Field | Value |
| --- | --- |
| **Mechanism** | `isinstance(audio_outputs, tuple)` |
| **Purpose** | In v4, the audio tower returns a tuple `(encodings, mask)`. In v5, it returns a named object with `.last_hidden_state` and `.audio_mel_mask`. |
| **v5-only** | Use only `audio_outputs.last_hidden_state` and `audio_outputs.audio_mel_mask`. Remove the `isinstance` tuple check. |

### 2.8 `vllm/model_executor/models/plamo3.py`

#### Shim — RoPE parameters nested vs flat dict (lines 162-174)

| Field | Value |
| --- | --- |
| **Mechanism** | `layer_type in config.rope_parameters` |
| **Purpose** | Same pattern as `gemma3.py` — handles v5 nested and v4 flat rope config. |
| **v5-only** | Access `config.rope_parameters[layer_type]` directly. Remove flat-dict fallback. |

### 2.9 `vllm/model_executor/models/modernbert.py`

#### Shim — Layer types and RoPE config (lines 87-106)

| Field | Value |
| --- | --- |
| **Mechanism** | `getattr(config, "layer_types", None)` |
| **Purpose** | In v5, `config.layer_types` and per-layer-type `rope_parameters` are available. In v4, layer type is inferred from `config.global_attn_every_n_layers` and rope params are constructed from `global_rope_theta`/`local_rope_theta`. |
| **v5-only** | Use `config.layer_types` and `config.rope_parameters[layer_type]` directly. Remove the v4 inference logic. |

### 2.10 `vllm/model_executor/models/ultravox.py`

#### Shim A — `layer_head_mask` argument (lines 400-404)

| Field | Value |
| --- | --- |
| **Mechanism** | `inspect.signature` check |
| **Purpose** | In v4, `WhisperEncoderLayer.forward` required `layer_head_mask` as a positional argument. In v5, it was removed. The shim passes it only when the signature requires it. |
| **v5-only** | Remove the `inspect.signature` check and `kwargs` dict. Call `layer(hidden_states, attention_mask=...)` directly. |

#### Shim B — `layer_head_mask` argument (second occurrence, lines 507-511)

| Field | Value |
| --- | --- |
| **Mechanism** | Same as Shim A |
| **Purpose** | Same pattern in a different Whisper encoder class. |
| **v5-only** | Same as Shim A. |

### 2.11 `vllm/model_executor/models/parakeet.py`

#### Shim — `convolution_bias` propagation (lines 116-123)

| Field | Value |
| --- | --- |
| **Mechanism** | Comment-documented behavioral difference |
| **Purpose** | In v5, `convolution_bias=False` is propagated from parakeet config causing `torch.conv1d` to skip registering the bias param. The method `_can_skip_missing_named_param` allows mismatched weights when bias tensors exist in the checkpoint but the module doesn't register them. |
| **v5-only** | Keep as-is — this handles weight loading for checkpoints that may have extra bias tensors. The logic is v5-aware, not a v4 shim. |

### 2.12 `vllm/model_executor/models/kimi_k25.py`

#### Shim — Token ID resolution (lines 116-124)

| Field | Value |
| --- | --- |
| **Mechanism** | Runtime comparison of tokenizer-resolved vs config IDs |
| **Purpose** | v5 may remap token IDs vs `config.json`. Resolves the ID from the tokenizer and uses the resolved value if it differs from the config. |
| **v5-only** | Keep — this is defensive code that works correctly on v5 and handles discrepancies. Could simplify by always using tokenizer-resolved IDs. |

### 2.13 `vllm/model_executor/model_loader/weight_utils.py`

#### Shim — Download acceleration (lines 98-103)

| Field | Value |
| --- | --- |
| **Mechanism** | `hasattr(huggingface_hub.constants, "HF_XET_HIGH_PERFORMANCE")` |
| **Purpose** | v5 (with newer `huggingface_hub`) supports `HF_XET_HIGH_PERFORMANCE` for fast downloads. v4 uses `hf_transfer` instead. |
| **v5-only** | Keep only `enable_xet_high_performance()`. Remove `enable_hf_transfer()` and the `hasattr` check. |

### 2.14 `vllm/model_executor/model_loader/gguf_loader.py`

#### Shim — Outer `model.` prefix stripping (lines 268-274)

| Field | Value |
| --- | --- |
| **Mechanism** | Prefix stripping with comment |
| **Purpose** | In v5, multimodal models wrap sub-models under an outer `model.` attribute, producing different state dict keys. This strips the prefix so keys match what gguf-py expects. |
| **v5-only** | Keep — this handles the v5 model structure. The prefix stripping is the correct v5 behavior. |

### 2.15 Weight prefix mappings (v4.52+ checkpoint names)

These files all contain `orig_to_new_prefix` entries in `WeightsMapper` to handle
checkpoint name changes introduced after Transformers v4.52. These are **not
runtime version branches** — they handle loading older checkpoints regardless
of the installed Transformers version. **Keep all of them.**

| File | Line |
| --- | --- |
| `vllm/model_executor/models/qwen2_5_vl.py` | 1134 |
| `vllm/model_executor/models/qwen2_vl.py` | 1135 |
| `vllm/model_executor/models/paligemma.py` | 271 |
| `vllm/model_executor/models/mistral3.py` | 377 |
| `vllm/model_executor/models/llava_next.py` | 230 |
| `vllm/model_executor/models/llava_next_video.py` | 304 |
| `vllm/model_executor/models/llava_onevision.py` | 484 |
| `vllm/model_executor/models/llava.py` | 519 |
| `vllm/model_executor/models/gemma3_mm.py` | 486 |
| `vllm/model_executor/models/gemma3n_mm.py` | 486 |
| `vllm/model_executor/models/aya_vision.py` | 318 |
| `vllm/model_executor/models/aria.py` | 508 |
| `vllm/model_executor/models/hunyuan_vision.py` | 810 |
| `vllm/model_executor/models/jina_vl.py` | 89 |
| `vllm/model_executor/models/mimo_v2_omni.py` | 1179 |

---

## 3 — `vllm/config/`, `vllm/renderers/`, `vllm/tokenizers/`

### 3.1 `vllm/config/vllm.py`

#### Shim — `tie_word_embeddings` propagation (lines 606-638)

| Field | Value |
| --- | --- |
| **Mechanism** | `Version(version("transformers")) >= Version("5.0.0")` |
| **Purpose** | In v5, `tie_word_embeddings` belongs to the composite model config (e.g. `SomeVLConfig`), not the text config. For multimodal models, this copies the value from `hf_config` to the language model's text config. |
| **v5-only** | Keep the propagation logic but remove the version check — always propagate for multimodal models. |

### 3.2 `vllm/config/model.py`

#### Shim — `rope_parameters` normalization (lines 2117-2121)

| Field | Value |
| --- | --- |
| **Mechanism** | Comment + `is_rope_parameters_nested()` check |
| **Purpose** | In v5, `rope_parameters` can be either `TypedDict` (flat) or `dict[str, TypedDict]` (nested by layer type). Normalizes to the nested form for uniform downstream processing. |
| **v5-only** | Keep — this normalization handles both rope_parameters shapes which can occur on v5. |

### 3.3 `vllm/renderers/hf.py`

#### Shim — `return_dict` default in `apply_chat_template` (lines 658-664)

| Field | Value |
| --- | --- |
| **Mechanism** | Conditional `return_dict=False` insertion |
| **Purpose** | v5 changed the default of `return_dict` to `True`, making `apply_chat_template(tokenize=True)` return a `BatchEncoding` instead of `list[int]`. Forces `return_dict=False` for consistency. |
| **v5-only** | Keep — this explicitly sets `return_dict=False` for downstream code that expects `list[int]`. It's v5-compatible and avoids depending on the default. |

### 3.4 `vllm/tokenizers/mistral.py`

#### Shim — `MistralCommonBackend` import (lines 40-47)

| Field | Value |
| --- | --- |
| **Mechanism** | `try/except ImportError` |
| **Purpose** | In v5, the class was renamed from `MistralCommonTokenizer` to `MistralCommonBackend` in `transformers.tokenization_mistral_common`. |
| **v5-only** | Direct import: `from transformers.tokenization_mistral_common import MistralCommonBackend`. Remove the `try/except`. |

---

## 4 — Test files

Test files contain version checks primarily for skipping tests that are
incompatible with specific Transformers versions. These should be reviewed
during the migration but are **lower priority** since they gate test execution
rather than production code.

### 4.1 `tests/models/registry.py`

Contains `min_transformers_version` and `max_transformers_version` fields in
`_HfExamplesInfo` entries. Key entries:

| Model | Constraint | Reason |
| --- | --- | --- |
| `Gemma4ForCausalLM` | `min>=5.0.0` | Requires v5 |
| `Glm4MoeLiteForCausalLM` | `min>=5.0.0` | Requires v5 |
| `Lfm2MoeForCausalLM` | `min>=5.0.0` | Requires v5 |
| `GlmAsrForConditionalGeneration` | `min>=5.0.0` | Requires v5 |
| `Lfm2VlForConditionalGeneration` | `min>=5.0.0` | Requires v5 |
| `Glm4MoeLiteMTPModel` | `min>=5.0.0` | Requires v5 |
| All `Transformers*` backend models | `min>=5.0.0` | Transformers backend requires v5 |
| `Plamo2ForCausalLM` | `max<=4.57` | `_tied_weight_keys` changed in v5 |
| `InternLM2VEForCausalLM` | `max<=4.57` | Custom config issues with v5 |
| `XverseForCausalLM` | `max<=4.57` | Tokenizer incompatible with v5 |
| Various MiniCPM models | `max<=4.57` | Custom processor code incompatible |
| Various InternVL models | `max<=4.57` | Custom code incompatible |

**v5-only:** Remove all `max_transformers_version` constraints (models capped at v4 would need upstream fixes or removal). Remove `min_transformers_version="5.0.0"` entries as the floor is now v5.

### 4.2 Other test files with version checks

| File | Mechanism | Purpose |
| --- | --- | --- |
| `tests/models/test_transformers.py:76-82` | `Version` check, `pytest.skip` | Skips OLMoE on v4 (MoE requires v5) |
| `tests/v1/e2e/spec_decode/test_spec_decode.py:391-397` | `Version` check, `pytest.skip` | Skips Eagle3 on v4 |
| `tests/v1/e2e/spec_decode/test_spec_decode.py:780-785` | `Version` check, `pytest.skip` | Skips Gemma4 MTP on <5.8.0 |
| `tests/models/quantization/test_bitsandbytes.py:143-148` | `pytest.mark.skipif` | Skips MoE bnb quantization on v5 |
| `tests/models/multimodal/generation/test_common.py:215-218` | `pytest.mark.skip` | Skips model incompatible with v5 |
| `tests/models/multimodal/generation/test_common.py:890-898` | `pytest.mark.skipif` | Skips model with v4.57.3 bug and v5 ROPE removal |
| `tests/models/multimodal/processing/test_musicflamingo.py:127-131` | `pytest.mark.skipif` | Skips on v5.5+ (native model added) |
| `tests/models/multimodal/generation/test_phi4siglip.py:24-28` | `pytest.mark.skipif` | Skips on v5+ (siglip2 changes) |
| `tests/lora/test_minicpmv_tp.py:16-22` | `pytest.mark.skipif` | Skips on v5+ (tokenizer.im_start_id unavailable) |
| `tests/lora/test_qwenvl.py:29-33` | `Version` check | Works around v4 Qwen2VLProcessor bug |
| `tests/renderers/test_chat_utils_prompt_embeds.py:501-503,560-562` | `return_dict=False` | Explicitly passes `return_dict=False` for v5 |
| `tests/entrypoints/serve/disagg/test_serving_tokens.py:197,239-240,268` | `return_dict=True`, `getattr_iter` | Uses v5 defaults and handles renamed attrs |
| `tests/entrypoints/offline_mode/test_offline_mode.py:111-113` | Aliased module pattern exclusion | v5 aliases modules that can't be reloaded |
| `tests/models/multimodal/pooling/test_phi3v.py:16-18` | Monkey-patch | BC for method deleted in v5 |
| `tests/models/multimodal/pooling/test_intern_vit.py:15-18` | `pytest.mark.skip` | Skip — custom code incompatible with v5 |
| `tests/models/multimodal/pooling/test_jinavl_reranker.py:18-21` | `pytest.mark.skip` | Skip — custom code incompatible with v5 |
| `tests/models/multimodal/pooling/test_colqwen3.py:25-28` | `pytest.mark.skip` | Skip — weight tying incompatible with v5 |
| `tests/models/multimodal/generation/test_nemotron_parse.py:106-109` | `pytest.mark.skip` | Skip — head count mismatch in v5 |
| `tests/models/multimodal/generation/test_voxtral.py:152-155` | `pytest.mark.skip` | Skip — `apply_chat_template` broken in v5 |
| `tests/models/language/pooling_mteb_test/test_jina.py:96-99` | `pytest.mark.skip` | Skip — custom code incompatible with v5 |
| `tests/models/language/pooling_mteb_test/test_gte.py:75-76` | `enable_test=False` | Skip — numerical regression with v5 |
| `tests/models/language/pooling_mteb_test/test_baai.py:72-75` | `enable_test=False` | Skip — custom tokenizer incompatible with v5 |

---

## Summary Statistics

| Category | Count |
| --- | --- |
| Source files with runtime v4/v5 branching | **27** |
| Distinct shims in source files | **35** |
| Test files with version-dependent logic | **18** |
| Weight prefix mappings (checkpoint compat, keep) | **15 files** |
| Shims safe to remove when v4 is dropped | **~25** |
| Shims to keep (v5-correct or checkpoint compat) | **~10** |

### Shim categories

1. **`try/except ImportError`** (2 shims): `config.py` ALLOWED_LAYER_TYPES, `mistral.py` MistralCommonBackend
2. **`Version()` comparison** (5 shims): `config.py` deprecation + RoPE, `base.py` weight renaming, `vllm.py` tie_word_embeddings, `weight_utils.py` xet/hf_transfer
3. **`hasattr` feature detection** (8 shims): `processor.py` (2), `pixtral.py`, `voxtral.py`, `deepseek_vl2.py`, `olmo_hybrid.py`, `qwen3_5.py`, `qwen3_next.py`, `qwen3_5_moe.py`, `speculators/base.py`
4. **Dict key / isinstance checks** (5 shims): `gemma3.py`, `gemma3n.py`, `plamo3.py`, `modernbert.py`, `gemma3n_mm.py` audio output
5. **`inspect.signature` checks** (2 shims): `ultravox.py` (2 locations)
6. **`check_version()` gates** (3 shims): `moe.py`, `base.py` encoder, `multimodal.py`
7. **Comment-documented workarounds** (3 shims): `qwen3_5.py`, `qwen3_5_moe.py` post-super init, `moe.py` fused experts
8. **Style name aliases** (1 shim): `utils.py` parallel style names

<!-- markdownlint-enable MD013 MD033 -->
