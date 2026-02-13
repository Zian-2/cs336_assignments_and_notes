import torch
print("-" * 30)
available = torch.cuda.is_available()
print(f"CUDA Available: {available}")
if available:
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("GPU not detected by PyTorch.")
print("-" * 30)