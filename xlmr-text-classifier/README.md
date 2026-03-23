# Fine-Tune XLM-RoBERTa for Text Classification

A Colab-ready notebook for fine-tuning [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) (or `xlm-roberta-large`) for text classification in any language. Supports both **single-text** and **text-pair** inputs.

## Features

- **Multilingual** — XLM-R supports 100+ languages out of the box
- **Single-text or text-pair** — sentiment, NLI, QE, similarity, etc.
- **Any number of classes** — binary or multi-class
- **Class balancing** — optional oversampling and/or class-weighted loss for imbalanced data
- **Stratified splits** — preserves label distribution across train/val/test
- **Detailed evaluation** — accuracy, F1, AUROC (binary), confusion matrix, classification report
- **Easy inference** — includes HuggingFace `pipeline` example for quick predictions

## Quick Start

1. **Prepare your data** as a CSV:

   **Single-text** (e.g., sentiment):
   | text | label |
   |------|-------|
   | This movie was great! | positive |
   | Terrible waste of time. | negative |

   **Text-pair** (e.g., NLI, quality estimation):
   | text | text_pair | label |
   |------|-----------|-------|
   | A man is playing guitar. | Someone is making music. | entailment |

2. **Open the notebook** in Colab: `xlmr_text_classifier.ipynb`

3. **Edit the config** (Section 3):
   - `data_path` — your CSV path
   - `label_names` — tuple of class names matching your `label` column
   - `text_pair_column` — set to your second column name, or `""` for single-text

4. **Run all cells**

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `xlm-roberta-base` | Base model (`xlm-roberta-large` for better accuracy) |
| `max_length` | 512 | Max tokens per input |
| `num_epochs` | 8 | Max training epochs (early stopping may stop sooner) |
| `train_batch_size` | 32 | Batch size (reduce if OOM) |
| `learning_rate` | 2e-5 | Learning rate |
| `oversample_minority` | False | Oversample minority classes to balance training |
| `use_class_weights` | False | Use inverse-frequency weights in loss |
| `early_stopping_patience` | 3 | Epochs without improvement before stopping |

## Output

After training, your `output_dir` will contain:

```
output_dir/
  checkpoints/          # training checkpoints
  best_model/           # fine-tuned model (ready for inference)
  test_predictions.csv  # per-sample predictions with probabilities
  results.json          # metrics, config, confusion matrix
```

## Requirements

- Python 3.8+
- GPU with >= 16GB VRAM (Colab T4 works)
- Dependencies: `transformers`, `torch`, `pandas`, `scikit-learn`, `evaluate`
