# build-gpt

My code-along of Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), trained on my MacBook (M3, 16 GB) with PyTorch's `mps` device. No NVIDIA GPU needed.

## Plan
1. **Code-along** on Tiny Shakespeare (character-level), matching the video's results.
2. **Tang poetry GPT (唐诗):** train the same model on ~55,000 Tang poems from [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) (MIT license). Character-level fits Chinese naturally: one character = one token, but the vocabulary grows from 65 to several thousand. Measure what it learns: correct 5- or 7-character lines? Matching line counts (绝句 = 4 lines, 律诗 = 8)? Rhyme?
3. **KV cache:** add one to generation and measure the speedup.
4. **Quantize the KV cache** to 8-bit and 4-bit and measure the quality loss.

Every experiment starts with a written prediction.

## Setup (already done)
- Python 3.12 virtual environment in `.venv/` (`torch`, `numpy`, `matplotlib`, `ipykernel`; see `requirements.txt`)
- Dataset: `input.txt`, Tiny Shakespeare (~1.1 MB) from [karpathy/char-rnn](https://github.com/karpathy/char-rnn), downloaded in the notebook with `!wget` like the video

## How to start
1. Open this folder in VS Code: `code projects/build-gpt` (or File → Open Folder).
2. Open `gpt.ipynb`. If VS Code asks for a kernel, pick **`.venv` (Python 3.12)**.
3. Run the first cell. It should print `device: mps`.

Check the setup anytime: `.venv/bin/python check_setup.py`

## Files
| File | What |
|---|---|
| `gpt.ipynb` | My code-along notebook: one section per part of the video |
| `bigram.py` | The bigram model as one script, with train/val loss estimates (`python bigram.py`, ~7 s on CPU, val loss ≈ 2.49) |
| `QUESTIONS.md` | Questions I write down while watching |
| `check_setup.py` | Confirms PyTorch + the Mac GPU + the dataset work |
