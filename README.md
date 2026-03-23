# AI Cookbooks

A collection of ready-to-use Colab notebooks for fine-tuning and experimenting with AI models.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [NLLB-600M Fine-Tune with LoRA](nllb-600m-finetune-lora/) | Fine-tune NLLB-200-distilled-600M for any translation task using LoRA & Flash Attention |
| [Gemma 3 1B Fine-Tune with LoRA](gemma3-1b-finetune-lora/) | Fine-tune Gemma 3 1B-IT for any instruction-following task using LoRA & Unsloth |
| [Gemma 3 4B Fine-Tune with LoRA](gemma3-4b-finetune-lora/) | Fine-tune Gemma 3 4B-IT (vision layers frozen) with QLoRA & Unsloth |

## Getting Started

Each notebook lives in its own folder with a README explaining setup and usage. Open the `.ipynb` file in Google Colab and follow the instructions.

## Contributing

To add a new notebook:
1. Create a folder with a descriptive name
2. Write a jupytext-compatible `.py` file and convert with `jupytext --to ipynb <file>.py`
3. Add a `README.md` inside the folder
4. Update this table
