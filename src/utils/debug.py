import torch
import psutil


def check_memory():
    print(f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    print(f"RAM: {psutil.virtual_memory().percent}%")