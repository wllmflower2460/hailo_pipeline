import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Run a simple benchmark
    print("\nRunning simple benchmark...")
    
    # CPU benchmark
    start_time = time.time()
    a = torch.randn(5000, 5000)
    b = torch.randn(5000, 5000)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU benchmark
    start_time = time.time()
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("GPU not available. Using CPU only.")
    print("\nPossible reasons:")
    print("1. Your AMD Radeon 780M may not be fully supported by ROCm")
    print("2. Environment variables may not be correctly set")
    print("3. ROCm installation may be incomplete")
    print("\nSuggested workarounds:")
    print("1. Use CPU-only mode for PyTorch (it's still fast)")
    print("2. Try updating your system drivers")
    print("3. For production use, consider a more supported GPU")
