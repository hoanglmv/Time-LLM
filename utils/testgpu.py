import torch

if torch.cuda.is_available():
    print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
    print(f"Phiên bản CUDA: {torch.version.cuda}")
else:
    print("Đang sử dụng CPU")
