import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch.cuda.get_device_name(0)}")

a = torch.tensor([1., 2., 3.], device=device)
print("torch.square:", torch.square(a))  
print("a ** 2:      ", a ** 2)
print("a * a:       ", a * a)

# Benchmarking Function (Averaged over 100 runs)
def benchmark_cuda_fn(func, tensor, num_iters=100, num_warmup=10):
    for _ in range(num_warmup):
        _ = func(tensor)
    
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        _ = func(tensor)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters  # average time per run in ms

b = torch.randn(10000, 10000, device=device)

def square_2(x): return x ** 2
def square_3(x): return x * x

print("\n--- Accurate CUDA Event Timing (Average per call) ---")
print(f"torch.square: {benchmark_cuda_fn(torch.square, b):.4f} ms")
print(f"a ** 2:       {benchmark_cuda_fn(square_2, b):.4f} ms")
print(f"a * a:        {benchmark_cuda_fn(square_3, b):.4f} ms")

# Modern PyTorch Profiler
def profile_operation(func, tensor, name):
    print(f"\n--- Profiling: {name} ---")
    
    for _ in range(5):
        _ = func(tensor)
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        _ = func(tensor)
        torch.cuda.synchronize()
        
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

profile_operation(torch.square, b, "torch.square")
profile_operation(square_2, b, "a ** 2")
profile_operation(square_3, b, "a * a")