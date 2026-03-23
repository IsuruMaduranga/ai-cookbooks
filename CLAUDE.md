# AI Cookbooks

A collection of Colab-ready notebooks for fine-tuning and experimenting with AI models, published on GitHub for public use.

## Repo structure

Each notebook lives in its own folder:

```
<notebook-folder>/
  <notebook>.ipynb        # the published notebook (committed to git)
  <notebook>.py           # jupytext source (local only, gitignored)
  README.md               # usage docs for the notebook
```

## Workflow for adding/updating notebooks

1. Edit the jupytext `.py` source file (percent format)
2. Convert: `jupytext --to ipynb <file>.py`
3. Only commit the `.ipynb` and `README.md` — the `.py` source stays local
4. Update the table in the root `README.md`

## Conventions

- Notebooks should be self-contained and runnable on Google Colab with a single GPU
- Use a `@dataclass` config at the top so users can easily customize parameters
- Install dependencies in the first cell with `!pip install -q ...`
- Keep notebooks general-purpose — no hardcoded paths or project-specific logic
- Do not add Co-Authored-By lines to commit messages
