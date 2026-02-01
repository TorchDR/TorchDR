# TorchDR Development Notes

## Commits and PRs

- Do not include "Co-Authored-By: Claude" or similar in commit messages
- Use the torchdr conda env for `gh` CLI: `source /cv/data/regevlab/vanasseh/miniforge3/etc/profile.d/conda.sh && conda activate torchdr`

## Starting a New PR

To start a new PR from an up-to-date state:

```bash
git fetch upstream && git checkout main && git merge upstream/main && git push origin main
git checkout -b <branch-name>
```

## Building and Viewing Documentation Locally

On the Slurm AWS cluster, activate the torchdr conda environment and build/serve docs:

```bash
# Activate environment
source /cv/data/regevlab/vanasseh/miniforge3/etc/profile.d/conda.sh && conda activate torchdr
cd /cv/home/vanasseh/TorchDR/docs

# Build docs (fast - skip running examples)
make html SPHINXOPTS="-D plot_gallery=0"

# Build docs (full - runs examples, slower)
make html
```

Start the HTTP server in tmux:

```bash
tmux kill-session -t docs 2>/dev/null
tmux new-session -d -s docs "source /cv/data/regevlab/vanasseh/miniforge3/etc/profile.d/conda.sh && conda activate torchdr && cd /cv/home/vanasseh/TorchDR/docs && python -m http.server 8000 --directory build/html"
```

To view in browser, set up SSH port forwarding from your local machine:

```bash
ssh -L 8000:localhost:8000 <user>@<cluster-host>
```

Then open http://localhost:8000 in your browser.

To reattach to the docs session: `tmux attach -t docs`
