import torch
from FlagEmbedding import FlagReranker

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU 设备: {torch.cuda.get_device_name(0)}")
else:
    print("❌ 警告：未检测到可用 GPU，模型将回退到 CPU 运行。")