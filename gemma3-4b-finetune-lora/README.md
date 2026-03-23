# Fine-Tune Gemma 3 4B-IT with LoRA & Unsloth

A Colab-ready notebook for fine-tuning [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) on any instruction-following task using **LoRA** and **[Unsloth](https://github.com/unslothai/unsloth)** for 2x faster training.

## Why a separate notebook from the 1B?

Gemma 3 4B is a **multimodal model with vision layers**. The vision layers must be explicitly frozen (`finetune_vision_layers=False`) when fine-tuning for text-only tasks — simply changing the model name in the 1B notebook won't work correctly. This notebook also defaults to **4-bit quantization (QLoRA)** since the 4B model needs more VRAM.

## Features

- **Unsloth** — optimized training kernels, 2x faster with ~50% less memory
- **QLoRA by default** — 4-bit quantized base model fits on Colab T4/L4
- **Vision layers frozen** — only language layers are fine-tuned via LoRA
- **Chat template** — uses Gemma 3's native chat format with optional system prompt
- **Prompt masking** — loss is computed only on the response tokens
- **Early stopping** — stops when validation loss plateaus

## Quick Start

1. **Prepare your data** as a CSV with two columns:

   | instruction | response |
   |-------------|----------|
   | Explain quantum computing in simple terms. | Quantum computing uses quantum bits... |
   | ... | ... |

2. **Open the notebook** in Colab: `gemma3_4b_finetune_lora.ipynb`

3. **Edit the config** (Section 3) — set `data_path`, `output_dir`, and optionally `system_prompt`

4. **Run all cells**

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `unsloth/gemma-3-4b-it` | Unsloth-optimized Gemma 3 4B |
| `max_seq_length` | 2048 | Max tokens per example |
| `load_in_4bit` | True | QLoRA — set False if you have enough VRAM |
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `learning_rate` | 3e-5 | Learning rate |
| `num_train_epochs` | 5 | Max training epochs |
| `per_device_train_batch_size` | 32 | Batch size (reduce if OOM) |
| `system_prompt` | `""` | Optional system prompt for all examples |
| `subset_ratio` | 1.0 | Set < 1.0 for quick test runs |

## Output

After training, your `output_dir` will contain:

```
output_dir/
  checkpoints/          # training checkpoints
  final_model/          # LoRA adapter weights (ready for inference)
  train_log.jsonl       # training metrics per step
  test_predictions.csv  # test set predictions
```

## Requirements

- Python 3.8+
- GPU with >= 16GB VRAM (Colab T4 with 4-bit, A100/L4 for full precision)
- Dependencies: `unsloth`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`
