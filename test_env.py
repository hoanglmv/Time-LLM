import torch
import time

def test_gpu_pytorch():
    print(f"PyTorch version: {torch.__version__}")
    
    # Kiểm tra NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        print("✅ NVIDIA GPU is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    # Kiểm tra Apple Silicon (MPS)
    elif torch.backends.mps.is_available():
        print("✅ Apple Metal (MPS) is available!")
        device = torch.device("mps")
    else:
        print("❌ No GPU found. Using CPU.")
        device = torch.device("cpu")

    # Test hiệu năng đơn giản
    if device.type != 'cpu':
        print("\n--- Testing Performance ---")
        x = torch.randn(10000, 10000, device=device)
        y = torch.randn(10000, 10000, device=device)
        
        start = time.time()
        z = torch.matmul(x, y) # Phép nhân ma trận lớn
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.time()
        
        print(f"Matrix multiplication completed in: {end - start:.4f} seconds on {device}")

if __name__ == "__main__":
    test_gpu_pytorch()