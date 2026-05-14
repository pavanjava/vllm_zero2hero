import ollama

# ── 1. Simple single-turn inference ──────────────────────────────────────────
def simple_chat(prompt: str, model: str = "gemma3:latest") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ── 2. Streaming inference ────────────────────────────────────────────────────
def streaming_chat(prompt: str, model: str = "gemma3:latest") -> None:
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()


# ── 3. Multi-turn conversation ────────────────────────────────────────────────
history = [{"role": "system", "content": "You are a concise AI/ML assistant."}]

def multi_turn_chat(user_input: str, model: str = "gemma3:latest") -> str:
    history.append({"role": "user", "content": user_input})
    response = ollama.chat(model=model, messages=history)
    assistant_msg = response["message"]["content"]
    history.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg


# ── 4. Raw generate (no chat template) ───────────────────────────────────────
def raw_generate(prompt: str, model: str = "gemma3:latest") -> str:
    raw = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": 0.2, "num_predict": 128},
    )
    return raw["response"]


# ── 5. Embeddings ─────────────────────────────────────────────────────────────
def get_embeddings(text: str, model: str = "qwen3-embedding:4b") -> list:
    embed = ollama.embeddings(model=model, prompt=text)
    return embed["embedding"]


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ✅ Uncomment whichever you need and run

    print("Welcome to Ollama inference!")

    # --- Simple single-turn ---
    # result = simple_chat("Explain KV cache in 3 sentences.")
    # print(result)

    # --- Streaming ---
    # streaming_chat("Write about Explain KV cache and transformers.")

    # --- Multi-turn ---
    # print(multi_turn_chat("What is PagedAttention?"))
    # print(multi_turn_chat("How does it differ from standard KV cache?"))

    # --- Raw generate ---
    # result = raw_generate("def fibonacci(n):")
    # print(result)

    # --- Embeddings ---
    # vec = get_embeddings("Retrieval augmented generation with Qdrant")
    # print(f"Embedding dim: {len(vec)}")
