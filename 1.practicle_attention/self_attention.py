import time

import torch
import torch.nn.functional as F

# sample embeddings
embeddings = {
    'The': torch.tensor([0.592, 0.893, 0.3665, -0.1879, 0.6763, 0.8535,
                         0.8394, 0.7153, -0.781, 0.6006, -0.4392, -0.7847,
                         -0.02017, 0.1782, -0.12054, 0.9575, -0.74, 0.9575,
                         0.509, 0.04068, 0.799, 0.991, -0.963, 0.3901,
                         0.5054, 0.5625, 0.3909, -0.2454, -0.9434, 0.2001,
                         -0.69, 0.3604], dtype=torch.float16),
    'quick': torch.tensor([-0.351, 0.364, -0.4622, -0.785, -0.713, 0.3235,
                           0.805, -0.1052, -0.8813, 0.1123, 0.1168, 0.8477,
                           -0.335, 0.0728, 0.7974, -0.885, -0.794, 0.8726,
                           0.01779, -0.355, -0.2803, -0.8833, 0.4346, 0.7637,
                           0.2698, 0.414, 0.831, 0.2454, 0.0975, -0.6704,
                           0.2277, -0.5776], dtype=torch.float16),
    'brown': torch.tensor([-0.779, 0.3376, -0.8584, 0.2252, -0.03226, 0.9966,
                           -0.96, 0.9146, 0.5366, -0.4321, -0.6016, 0.797,
                           -0.531, -0.2379, -0.2295, 0.8643, 0.2998, 0.744,
                           0.4326, 0.4004, -0.0574, -0.149, -0.3262, 0.5386,
                           -0.771, 0.9766, -0.7666, 0.1707, -0.896, 0.8535,
                           0.4885, 0.13], dtype=torch.float16),
    'fox': torch.tensor([-0.09735, -0.2688, 0.2979, -0.398, -0.0605, 0.5034,
                         -0.82, 0.507, -0.3127, -0.498, -0.2896, 0.1781,
                         -0.917, -0.4133, 0.7964, 0.4768, -0.9375, 0.7847,
                         0.8247, -0.915, -0.5415, -0.4097, 0.2524, -0.541,
                         -0.7964, 0.4294, -0.3972, -0.2515, -0.05936, -0.654,
                         -0.6567, 0.577], dtype=torch.float16),
    'jumped': torch.tensor([0.5386, 0.0206, 0.0122, 0.2651, 0.3203, 0.2212,
                            0.849, -0.0855, 0.1162, 0.468, 0.8374, 0.599,
                            0.01012, 0.7534, 0.1228, -0.532, 0.409, -0.262,
                            -0.902, 0.02083, 0.4187, -0.758, 0.2369, 0.6978,
                            0.08594, 0.3054, -0.757, 0.2832, -0.5874, 0.7896,
                            0.02171, -0.8623], dtype=torch.float16)
}

d_model = 32
d_k = 32

W_Q = torch.empty(d_model, d_k, dtype=torch.float16).uniform_(-1, 1)
W_K = torch.empty(d_model, d_k, dtype=torch.float16).uniform_(-1, 1)
W_V = torch.empty(d_model, d_k, dtype=torch.float16).uniform_(-1, 1)

print("W_Q shape:", W_Q.shape)
print("W_K shape:", W_K.shape)
print("W_V shape:", W_V.shape)


# ── Simple Attention ───────────────────────────────────────────────────
def compute_attention(input_words):
    # Stack embeddings -> (seq_len, 64)
    E = torch.stack([embeddings[word] for word in input_words])

    # Compute K, V for all tokens
    K = E @ W_K  # (seq_len, d_k)
    V = E @ W_V  # (seq_len, d_k)

    # Compute Q for last token
    Q = E[-1] @ W_Q  # (d_k,)

    # Scaled dot-product scores
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    scores = (Q.float() @ K.float().T) / scale  # (seq_len,)

    # F.softmax handles numerical stability internally
    attention_weights = F.softmax(scores, dim=-1).half()  # (seq_len,)

    # Weighted sum of values
    output = attention_weights @ V  # (d_k,)

    return output


if __name__ == '__main__':
    # Step 1: predict "brown"
    output_step1 = compute_attention(['The', 'quick'])
    print("Step 1 output shape:", output_step1.shape, "\n", output_step1)

    # Step 2: predict "fox"
    output_step2 = compute_attention(['The', 'quick', 'brown'])
    print("Step 2 output shape:", output_step2.shape, "\n", output_step2)

    # Step 3: predict "jumped"
    start = time.time()
    output_step3 = compute_attention(['The', 'quick', 'brown', 'fox'])
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print("Step 3 output shape:", output_step3.shape, "\n", output_step3)
