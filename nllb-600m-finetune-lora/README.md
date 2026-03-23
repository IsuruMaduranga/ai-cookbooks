# Fine-Tune NLLB-200-distilled-600M with LoRA & Flash Attention

A Colab-ready notebook for fine-tuning [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) on any translation task using **LoRA** (parameter-efficient fine-tuning) and **SDPA / Flash Attention** for faster training.

## Features

- **LoRA fine-tuning** — trains only ~2% of parameters, runs on a free Colab T4 GPU
- **Flash Attention (SDPA)** — uses PyTorch's scaled dot-product attention for memory-efficient training
- **Automatic metrics** — evaluates with BLEU, TER, and chrF2
- **Early stopping** — stops training when validation TER stops improving
- **Ready-to-use inference** — includes code to translate new text with the fine-tuned model

## Quick Start

1. **Prepare your data** as a CSV with two columns:

   | source | target |
   |--------|--------|
   | Hello, how are you? | Bonjour, comment allez-vous? |
   | ... | ... |

2. **Open the notebook** in Colab: `nllb_600m_finetune_lora.ipynb`

3. **Edit the config** (Section 3) — set your `data_path`, `src_lang`, `tgt_lang`, and `output_dir`

4. **Run all cells**

## NLLB Language Codes

NLLB uses language codes like `eng_Latn`, `fra_Latn`, `sin_Sinh`, `zho_Hans`, etc.

Full list: [FLORES-200 language codes](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 32 | LoRA rank (higher = more capacity, more memory) |
| `lora_alpha` | 64 | LoRA scaling factor |
| `learning_rate` | 2e-4 | Learning rate |
| `num_train_epochs` | 8 | Max training epochs (early stopping may stop sooner) |
| `per_device_train_batch_size` | 48 | Batch size (reduce if OOM) |
| `use_flash_attention` | True | Enable SDPA attention |
| `subset_ratio` | 1.0 | Set < 1.0 for quick test runs |

## Output

After training, your `output_dir` will contain:

```
output_dir/
  checkpoints/       # training checkpoints
  final_model/       # LoRA adapter weights (ready for inference)
  train_log.jsonl    # training metrics per step
  test_predictions.csv  # test set translations
  test_metrics.json  # BLEU, TER, chrF2 scores
```

## Requirements

- Python 3.8+
- GPU with >= 16GB VRAM (Colab T4 works)
- Dependencies: `transformers`, `datasets`, `evaluate`, `sacrebleu`, `sentencepiece`, `peft`, `accelerate`
