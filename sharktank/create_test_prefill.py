import torch
import numpy as np

# softmax(Q * K^t) * V
if __name__ == "__main__":
    m = 4096
    k2 = 64
    n = 64
    b0 = 1
    b1 = 1
    q = (torch.rand(4, 64, 32, 128).to(torch.float32) - 0.5) * 1
    k = (torch.rand(4, 64, 32, 128).to(torch.float32) - 0.5) * 1
    v = (torch.rand(4, 64, 32, 128).to(torch.float32) - 0.5) * 1
    # mask = (torch.rand(4, 64, 32, 128).to(torch.float32) - 0.5) * 1

    np.save("attn_q.npy", q.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_k.npy", k.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("attn_v.npy", v.detach().to(dtype=torch.float16, device="cpu").numpy())
    np.save("arg3.npy", torch.zeros((4), dtype=torch.int64).cpu().numpy())
    np.save("arg4.npy", torch.zeros((4, 4), dtype=torch.int64).cpu().numpy())
    np.save("arg5.npy", torch.zeros((256, 4194304), dtype=torch.float16).cpu().numpy())
    print("go")
    # np.save("attn_mask.npy", mask.detach().to(dtype=torch.float16, device="cpu").numpy())

    # Post attention func
    out = torch._scaled_dot_product_flash_attention_for_cpu(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).output.transpose(1, 2)
    out = torch.flatten(out, start_dim=2, end_dim=3)
    np.save("attn_ref.npy", out.detach().to(device="cpu", dtype=torch.float16).numpy())