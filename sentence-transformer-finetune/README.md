# Fine-Tune a Sentence Transformer

A 3-notebook pipeline for fine-tuning any [sentence-transformers](https://www.sbert.net/) model on text-pair data and publishing it to HuggingFace Hub.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [01_filter_dataset.ipynb](01_filter_dataset.ipynb) | Download & filter a HuggingFace dataset by token length |
| 2 | [02_finetune.ipynb](02_finetune.ipynb) | Fine-tune the model with contrastive learning |
| 3 | [03_push_to_hub.ipynb](03_push_to_hub.ipynb) | Generate model card & push to HuggingFace Hub (+ optional ONNX) |

Notebook 1 is **optional** — you can bring your own data as a JSON file with `text_a` and `text_b` pairs.

## Use Cases

- **Code search** — query ↔ code snippet (e.g., CodeSearchNet)
- **Semantic search** — query ↔ document
- **Sentence similarity** — sentence A ↔ sentence B
- **FAQ matching** — question ↔ answer

## Data Format

The fine-tuning notebook expects a JSON file:

```json
[
  {"text_a": "function that adds two numbers", "text_b": "def add(a, b): return a + b"},
  {"text_a": "sort a list in python", "text_b": "sorted_list = sorted(my_list)"}
]
```

The filtering notebook (01) produces this format automatically from any HuggingFace dataset.

## How It Works

1. **MultipleNegativesRankingLoss** — each batch of N pairs provides N-1 hard negatives per sample automatically
2. **InformationRetrievalEvaluator** — evaluates with MRR, Accuracy@K, Recall@K, NDCG during training
3. **Base vs fine-tuned comparison** — automatically benchmarks improvement over the base model

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model |
| `MAX_SEQ_LENGTH` | 512 | Max tokens (filter + model config) |
| `BATCH_SIZE` | 192 | Larger = more in-batch negatives (reduce if OOM) |
| `EPOCHS` | 3 | Training epochs |
| `LEARNING_RATE` | 2.7e-5 | Learning rate |
| `EVAL_SAMPLES` | 2000 | Held-out pairs for evaluation |

## Output

All outputs are saved to Google Drive:

```
output_dir/
  trained_model/          # fine-tuned model weights
  training_metadata.json  # hyperparameters and timing
  eval_results.json       # fine-tuned model metrics
  base_eval_results.json  # base model metrics
  comparison.json         # side-by-side comparison
  checkpoints/            # intermediate checkpoints
```

## Requirements

- Python 3.8+
- GPU with >= 16GB VRAM (Colab T4 works for most models)
- Dependencies: `sentence-transformers`, `datasets`, `huggingface_hub`
