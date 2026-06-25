# Reinforcement Learning from Human Feedback

Reinforcement Learning from Human Feedback (RLHF) is a technique that fine-tunes language models using human-generated preference data to align model outputs with desired behaviors. vLLM can be used to generate the completions for RLHF.

## Prerequisites

Before integrating vLLM into an RL training loop, ensure:

- vLLM is installed with GPU support (see the main [installation guide](../getting_started/installation/index.md))
- The model you want to fine-tune is supported by vLLM (see [supported models](../models/supported_models.md))
- For weight synchronization between trainer and inference engine, review the [Weight Transfer](weight_transfer/README.md) backends (NCCL for multi-GPU, IPC for same-GPU)

## Integrations

The following open-source RL libraries use vLLM for fast rollouts (sorted alphabetically and non-exhaustive):

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)

For weight synchronization between training and inference, see the [Weight Transfer](weight_transfer/README.md) documentation, which covers the pluggable backend system with [NCCL](weight_transfer/nccl.md) (multi-GPU) and [IPC](weight_transfer/ipc.md) (same-GPU) engines.

For pipelining generation and training to improve GPU utilization and throughput, see the [Async Reinforcement Learning](async_rl.md) guide, which covers the pause/resume API for safely updating weights mid-flight.

See the following notebooks showing how to use vLLM for GRPO:

- [Efficient Online Training with GRPO and vLLM in TRL](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
