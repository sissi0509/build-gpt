"""Run once to confirm the environment works: .venv/bin/python check_setup.py"""
import torch

print("PyTorch:", torch.__version__)
print("Mac GPU (mps) available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(1000, 1000, device=device)
y = x @ x  # a matrix multiply on the chosen device
print("Matrix multiply on", device, "OK, result shape:", tuple(y.shape))

text = open("data/input.txt", encoding="utf-8").read()
print("Dataset characters:", len(text), "| unique characters:", len(set(text)))
