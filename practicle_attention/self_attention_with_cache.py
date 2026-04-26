import time

import torch
import torch.nn.functional as F

embeddings = {
    'The': torch.tensor([0.592, 0.893, 0.3665, -0.1879, 0.6763, 0.8535,
                         0.8394, 0.7153, -0.781, 0.6006, -0.4392, -0.7847,
                         -0.02017, 0.1782, -0.12054, 0.9575, -0.74, 0.9575,
                         0.509, 0.04068, 0.799, 0.991, -0.963, 0.3901,
                         0.5054, 0.5625, 0.3909, -0.2454, -0.9434, 0.2001,
                         0.1553, 0.0834, -0.934, 0.295, -0.2896, -0.646,
                         -0.2502, -0.9443, 0.6924, 0.4941, 0.9653, -0.1827,
                         -0.06885, 0.1318, -0.1372, 0.9424, -0.9116, -0.381,
                         -0.1351, 0.2673, -0.02303, 0.6665, -0.4026, 0.6133,
                         0.9097, 0.335, 0.6777, -0.9355, 0.0801, 0.01872,
                         0.653, 0.9375, -0.69, 0.3604], dtype=torch.float16),
    'quick': torch.tensor([-0.351, 0.364, -0.4622, -0.785, -0.713, 0.3235,
                           0.805, -0.1052, -0.8813, 0.1123, 0.1168, 0.8477,
                           -0.335, 0.0728, 0.7974, -0.885, -0.794, 0.8726,
                           0.01779, -0.355, -0.2803, -0.8833, 0.4346, 0.7637,
                           0.2698, 0.414, 0.831, 0.2454, 0.0975, -0.6704,
                           -0.605, -0.6406, -0.568, 0.5854, -0.851, 0.1766,
                           -0.767, 0.5366, -0.9814, 0.203, -0.164, -0.216,
                           0.2196, 0.404, -0.4893, -0.9185, -0.3237, -0.7812,
                           0.514, 0.663, -0.968, -0.28, -0.215, 0.6743,
                           0.916, -0.905, 0.5957, -0.3162, -0.202, 0.5044,
                           -0.628, 0.425, 0.2277, -0.5776], dtype=torch.float16),
    'brown': torch.tensor([-0.779, 0.3376, -0.8584, 0.2252, -0.03226, 0.9966,
                           -0.96, 0.9146, 0.5366, -0.4321, -0.6016, 0.797,
                           -0.531, -0.2379, -0.2295, 0.8643, 0.2998, 0.744,
                           0.4326, 0.4004, -0.0574, -0.149, -0.3262, 0.5386,
                           -0.771, 0.9766, -0.7666, 0.1707, -0.896, 0.8535,
                           0.0672, -0.527, 0.3208, 0.671, 0.7393, 0.1902,
                           0.1538, 0.8223, -0.8613, -0.916, 0.4993, -0.287,
                           0.6196, -0.9966, -0.4062, -0.0571, -0.6235, 0.8955,
                           -0.674, -0.7437, -0.2339, -0.845, 0.04916, 0.866,
                           0.788, -0.994, 0.493, 0.2683, -0.4458, -0.447,
                           0.5947, -0.1901, 0.4885, 0.13], dtype=torch.float16),
    'fox': torch.tensor([-0.09735, -0.2688, 0.2979, -0.398, -0.0605, 0.5034,
                         -0.82, 0.507, -0.3127, -0.498, -0.2896, 0.1781,
                         -0.917, -0.4133, 0.7964, 0.4768, -0.9375, 0.7847,
                         0.8247, -0.915, -0.5415, -0.4097, 0.2524, -0.541,
                         -0.7964, 0.4294, -0.3972, -0.2515, -0.05936, -0.654,
                         -0.8354, -0.5107, -0.5063, 0.2096, -0.3596, 0.1594,
                         0.6084, -0.5483, -0.131, 0.2952, 0.253, 0.5107,
                         -0.9297, -0.888, -0.3416, 0.11304, -0.425, -0.7026,
                         -0.198, 0.01964, 0.2754, 0.2566, -0.22, -0.08966,
                         0.772, -0.476, -0.896, 0.003637, 0.5845, -0.2086,
                         0.41, -0.1837, -0.6567, 0.577], dtype=torch.float16),
    'jumped': torch.tensor([0.5386, 0.0206, 0.0122, 0.2651, 0.3203, 0.2212,
                            0.849, -0.0855, 0.1162, 0.468, 0.8374, 0.599,
                            0.01012, 0.7534, 0.1228, -0.532, 0.409, -0.262,
                            -0.902, 0.02083, 0.4187, -0.758, 0.2369, 0.6978,
                            0.08594, 0.3054, -0.757, 0.2832, -0.5874, 0.7896,
                            0.358, -0.031, -0.4673, 0.3164, -0.505, 0.1644,
                            -0.1754, 0.69, 0.2415, -0.7734, 0.9253, 0.038,
                            0.996, 0.04663, 0.9077, -0.608, 0.2568, -0.3176,
                            -0.6377, -0.7046, 0.806, 0.93, 0.7666, -0.803,
                            0.264, -0.292, 0.3398, 0.03302, 0.1825, 0.4597,
                            0.2742, -0.04172, 0.02171, -0.8623], dtype=torch.float16)
}

d_model = 64
d_k = 64

W_Q = torch.empty(d_model, d_k, dtype=torch.float16).uniform_(-1, 1)
W_K = torch.empty(d_model, d_k, dtype=torch.float16).uniform_(-1, 1)
W_V = torch.empty(d_model, d_k, dtype=torch.float16).uniform_(-1, 1)

# ── KV Cache ──────────────────────────────────────────────────────────────────
kv_cache = {
    'keys':   [],   # list of (d_k,) tensors, one per cached token
    'values': [],   # list of (d_k,) tensors, one per cached token
    'words':  []    # tracks which words are already cached
}

def clear_cache():
    kv_cache['keys'].clear()
    kv_cache['values'].clear()
    kv_cache['words'].clear()
    print("Cache cleared.")

def cache_stats():
    print(f"Cached tokens : {kv_cache['words']}")
    print(f"Cache size    : {len(kv_cache['words'])} tokens")

# ── Attention with KV Cache ───────────────────────────────────────────────────
def compute_attention_with_cache(input_words):
    new_words = [w for w in input_words if w not in kv_cache['words']]
    cached_words = kv_cache['words'][:]

    print(f"\nInput       : {input_words}")
    print(f"Cache hit   : {cached_words}")
    print(f"Cache miss  : {new_words}  → computing K,V for these only")

    # Compute K, V only for new (uncached) tokens
    for word in new_words:
        e = embeddings[word]                    # (d_k,)
        k = e @ W_K                             # (d_k,)
        v = e @ W_V                             # (d_k,)
        kv_cache['keys'].append(k)
        kv_cache['values'].append(v)
        kv_cache['words'].append(word)

    # Assemble full K, V from cache (preserves sequence order)
    ordered_words = [w for w in input_words]    # keep original order
    key_list = []
    val_list = []
    for w in ordered_words:
        idx = kv_cache['words'].index(w)
        key_list.append(kv_cache['keys'][idx])
        val_list.append(kv_cache['values'][idx])

    K = torch.stack(key_list)                   # (seq_len, d_k)
    V = torch.stack(val_list)                   # (seq_len, d_k)

    # Q is computed fresh for the last (query) token every time
    Q = embeddings[input_words[-1]] @ W_Q       # (d_k,)

    # Scaled dot-product attention
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    scores = (Q.float() @ K.float().T) / scale  # (seq_len,)
    attention_weights = F.softmax(scores, dim=-1).half()
    output = attention_weights @ V               # (d_k,)

    return output


if __name__ == '__main__':
    # Step 1: predict "brown" — computes K,V for [The, quick]
    out1 = compute_attention_with_cache(['The', 'quick'])
    cache_stats()

    # Step 2: predict "fox" — reuses [The, quick], computes only "brown"
    out2 = compute_attention_with_cache(['The', 'quick', 'brown'])
    cache_stats()

    # Step 3: predict "jumped" — reuses [The, quick, brown], computes only "fox"
    start = time.time()
    out3 = compute_attention_with_cache(['The', 'quick', 'brown', 'fox'])
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    cache_stats()

    print("\n--- Outputs ---")
    print("Step 1:", out1.shape, out1)
    print("Step 2:", out2.shape, out2)
    print("Step 3:", out3.shape, out3)