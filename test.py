import torch
from torch import nn

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Is CUDA available:", torch.cuda.is_available())

def dump(x, name):
    print(name, "device:", x.device, "dtype:", x.dtype, "shape:", tuple(x.shape), "contig:", x.is_contiguous())

# 构造最小例子
B, C = 4, 512
q = torch.randn(B, 1, C, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, 1, C, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, 1, C, device='cuda', dtype=torch.bfloat16)

mha = nn.MultiheadAttention(embed_dim=C, num_heads=1, batch_first=True).to('cuda', dtype=torch.bfloat16)

dump(q, "q"); dump(k, "k"); dump(v, "v")
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
    out, _ = mha(q, k, v, need_weights=False)
print("OK, out:", out.shape)