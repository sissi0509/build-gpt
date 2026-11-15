# build-gpt

My code-along of Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), trained on my MacBook (M3, 16 GB) with PyTorch's `mps` device. No NVIDIA GPU needed.

Next steps after the video: add a KV cache and measure the speedup, then quantize the KV cache to 8-bit and 4-bit and measure the quality loss.

## Setup (already done)
- Python 3.12 virtual environment in `.venv/` (`torch`, `numpy`, `matplotlib`, `ipykernel`; see `requirements.txt`)
- Dataset: `data/input.txt`, Tiny Shakespeare (~1.1 MB) from [karpathy/char-rnn](https://github.com/karpathy/char-rnn)

## How to start
1. Open this folder in VS Code: `code projects/build-gpt` (or File → Open Folder).
2. Open `gpt.ipynb`. If VS Code asks for a kernel, pick **`.venv` (Python 3.12)**.
3. Run the first cell. It should print `device: mps`.

Check the setup anytime: `.venv/bin/python check_setup.py`

## Files
| File | What |
|---|---|
| `gpt.ipynb` | My code-along notebook: one section per part of the video, cells empty for me to fill |
| `QUESTIONS.md` | Questions I write down while watching |
| `check_setup.py` | Confirms PyTorch + the Mac GPU + the dataset work |
