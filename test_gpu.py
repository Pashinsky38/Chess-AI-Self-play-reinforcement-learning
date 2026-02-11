import torch
import time

print("="*60)
print("GPU Test for Chess AI")
print("="*60)

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Speed test
    print("\n" + "="*60)
    print("Speed Test: CPU vs GPU")
    print("="*60)
    
    # Create test tensor
    size = 1000
    
    # CPU test
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    start = time.time()
    for _ in range(10):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU time (10 iterations): {cpu_time:.4f} seconds")
    
    # GPU test
    x_gpu = torch.randn(size, size).cuda()
    y_gpu = torch.randn(size, size).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU time (10 iterations): {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x faster")
    
    print("\n✅ GPU is working! Your Chess AI will train much faster.")
    print("\nNow run your Chess AI:")
    print("  python chess_ai_selfplay_improved.py")
    print("\nCheck the statistics panel - it should show: Device: cuda")
else:
    print("\n❌ GPU not available. Chess AI will use CPU (slower).")
    print("\nTroubleshooting:")
    print("1. Check if you have an NVIDIA GPU")
    print("2. Install/update NVIDIA drivers")
    print("3. Install PyTorch with CUDA support")
    print("\nQuick fix - try running:")
    print("  pip uninstall torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("="*60)
