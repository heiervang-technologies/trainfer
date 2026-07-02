import torch
import torch.nn.functional as F
import time

def benchmark_kl(L, V):
    t_logits = torch.randn(L, V, device="cuda", requires_grad=True)
    s_logits = torch.randn(L, V, device="cuda", requires_grad=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    t_log_p = F.log_softmax(t_logits.float(), dim=-1)
    s_log_p = F.log_softmax(s_logits.float(), dim=-1)
    p = torch.exp(t_log_p)
    kl = (p * (t_log_p - s_log_p)).sum(dim=-1).mean()
    kl.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"Original: {elapsed*1000:.2f} ms, Peak Mem: {peak_mem:.2f} MB")

    # Chunked
    t_logits.grad = None
    s_logits.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    kl_sum = 0.0
    chunk_size = 128
    for start_idx in range(0, L, chunk_size):
        end_idx = min(start_idx + chunk_size, L)
        t_chunk = t_logits[start_idx:end_idx].float()
        s_chunk = s_logits[start_idx:end_idx].float()
        t_log_p = F.log_softmax(t_chunk, dim=-1)
        s_log_p = F.log_softmax(s_chunk, dim=-1)
        p = torch.exp(t_log_p)
        chunk_kl = (p * (t_log_p - s_log_p)).sum(dim=-1)
        kl_sum += chunk_kl.sum()
    kl2 = kl_sum / L
    kl2.backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"Chunked: {elapsed*1000:.2f} ms, Peak Mem: {peak_mem:.2f} MB")
    print(f"KL match: {torch.allclose(kl, kl2)}")

benchmark_kl(512, 152000)
