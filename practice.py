# %%
"""
PyTorch Tensor Reps — Set 1 (no solutions)

Rules:
- Do each cell fast. Aim 2–5 min each.
- Use torch + einops. Prefer vectorized ops.
- Don’t write Python loops unless the prompt says you can.

Assume device='mps' (Mac).
"""
import torch as t
import einops
device = "mps"
t.manual_seed(0)
# %%
# %%
# Problem 1: Basic rearrange
# Given x with shape (B, T, C). Create y with shape (B*T, C) without copying if possible.
B, T_, C = 4, 7, 16
x = t.randn(B, T_, C, device=device)
y = einops.rearrange(x, 'b t c -> (b t) c')
# TODO: y = ...
assert y.shape == (B*T_, C)
# %%
# %%
# Problem 2: Split + merge heads (classic attention prep)
# Given x: (B, T, D) with D = H * Dh, create (B, H, T, Dh).
B, T_, H, Dh = 2, 5, 4, 8
D = H * Dh
x = t.randn(B, T_, D, device=device)
# TODO: x_heads = ...
# assert x_heads.shape == (B, H, T_, Dh)
x_heads = einops.rearrange(x, 'b t (h dh) -> b h t dh', h=H)
assert x_heads.shape == (B, H, T_, Dh)
# %%
# Problem 3: Reverse of Problem 2
# Given x_heads: (B, H, T, Dh), return x: (B, T, H*Dh).
B, H, T_, Dh = 2, 4, 5, 8
x_heads = t.randn(B, H, T_, Dh, device=device)
x = einops.rearrange(x_heads, 'b h t dh -> b t (h dh)')
assert x.shape == (B, T_, H*Dh)
# %%
# Problem 4: Gather Q(s,a) for a batch (DQN staple)
# q: (B, A), actions: (B,) int64 -> qsa: (B,)
B, A = 6, 9
q = t.randn(B, A, device=device)
actions = t.randint(0, A, (B,), device=device, dtype=t.int64)
qsa = t.gather(q, dim=1, index=actions.unsqueeze(1)).squeeze(1)
# TODO: qsa = ...
assert qsa.shape == (B,)

# %%
# %%
# Problem 5: Gather with extra dims (sequence setting)
# q: (B, T, A), actions: (B, T) -> qsa: (B, T)
B, T_, A = 3, 5, 7
q = t.randn(B, T_, A, device=device)
actions = t.randint(0, A, (B, T_), device=device, dtype=t.int64)
qsa = t.gather(q, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
# TODO: qsa = ...
assert qsa.shape == (B, T_)
# %%
# %%
# Problem 6: Mask illegal actions in logits
# logits: (B, A), legal: (B, A) bool. Set logits[~legal] = -1e9 (or very negative), keep dtype/device.
B, A = 4, 10
logits = t.randn(B, A, device=device)
legal = t.rand(B, A, device=device) > 0.3
masked_logits = t.where(legal, logits, -1e9)
# TODO: masked_logits = ...
assert masked_logits.shape == (B, A)
assert t.isfinite(masked_logits[legal]).all()
# %%
# %%
# Problem 7: Stable log-softmax + gather log-prob of chosen action
# logits: (B, A), actions: (B,) -> logp: (B,)
B, A = 8, 6
logits = t.randn(B, A, device=device)
logits -= logits.max(dim=1, keepdim=True).values
probs = t.exp(logits) 
probs /= probs.sum(dim=1, keepdim=True)
logp = t.log(probs)
assert(logp.shape == (B, A))
actions = t.randint(0, A, (B,), device=device, dtype=t.int64)
logp_a = t.gather(input=logp, dim=1, index=actions.unsqueeze(1)).squeeze(1)
# TODO: logp = ...
# Hint: use t.log_softmax then gather
assert logp_a.shape == (B,)
# %%
# %%
# Problem 8: Compute entropy from logits (no loops)
# logits: (B, A) -> entropy: (B,)
B, A = 5, 11
logits = t.randn(B, A, device=device)
logits -= logits.max(dim=1, keepdim=True).values
probs = t.exp(logits) 
probs /= probs.sum(dim=1, keepdim=True)
logp = t.log(probs)
plp = -probs * logp
entropy = plp.sum(dim=1)
# TODO: entropy = ...
assert entropy.shape == (B,)
# entropy should be >= 0
# %%
# %%
# Problem 9: Broadcasty advantage normalization (PPO staple)
# adv: (T, B) -> normalize over all entries (single mean/std), keep shape (T,B)
T_, B = 12, 4
adv = t.randn(T_, B, device=device)
adv_norm = (adv - adv.mean())/(adv.std() + 1e-8)
# TODO: adv_norm = ...
assert adv_norm.shape == (T_, B)
assert abs(float(adv_norm.mean())) < 1e-3
assert abs(float(adv_norm.std() - 1.0)) < 1e-2
# %%
# %%
# Problem 10: Turn a flat batch into episodes via lengths (ragged → padded)
# You get a flat tensor x: (N, D) and lengths: (E,) summing to N.
# Build x_padded: (E, Lmax, D) and mask: (E, Lmax) where mask=True indicates real token.
E = 4
lengths = t.tensor([3, 7, 2, 5], device=device)
N = int(lengths.sum().item())
D = 6
x = t.randn(N, D, device=device)
Lmax = int(lengths.max().item())
x_padded = t.zeros(size=(E, Lmax, D), device=x.device, dtype=x.dtype)
mask = t.zeros(size=(E, Lmax), device=x.device, dtype=t.bool)
start_index = 0
for i in range(E):
    x_padded[i, :lengths[i]] = x[start_index:start_index+lengths[i].item(), :]
    start_index += lengths[i].item()
    mask[i, :lengths[i].item()] = True


# TODO: x_padded, mask = ...
assert x_padded.shape[0] == E and x_padded.shape[2] == D
assert mask.shape == (E, int(lengths.max().item()))
# %%
