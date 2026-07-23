"""
验证GPU是否可用
"""
import torch

print("=" * 60)
print("GPU Verification Test")
print("=" * 60)

print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"2. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. CUDA Version: {torch.version.cuda}")
    print(f"4. GPU Count: {torch.cuda.device_count()}")
    print(f"5. Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"6. GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试实际使用GPU
    print("\n7. Testing GPU computation...")
    try:
        x = torch.randn(2, 3, 224, 224).cuda()
        y = x * 2
        print(f"   [OK] GPU computation test successful!")
        print(f"   [OK] Tensor device: {x.device}")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] GPU is available and working!")
        print("=" * 60)
        
    except Exception as e:
        print(f"   [ERROR] GPU test failed: {e}")
        print("\n" + "=" * 60)
        print("[FAIL] GPU may not be usable, will use CPU")
        print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("[FAIL] CUDA not detected, will use CPU")
    print("=" * 60)
    print("\nTo install GPU version of PyTorch:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
