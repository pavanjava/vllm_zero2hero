import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available. Please ensure you have a GPU runtime selected.")

from huggingface_hub import login
from vllm import LLM, SamplingParams

login()

prompts = [
    "Hi, my name is ....",
    "Today is a beautiful summer day ...",
    "Hello there",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
)

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

outputs = llm.generate(prompts, sampling_params=sampling_params)

for i, output in enumerate(outputs):
  print(f"Prompt: {prompts[i]}")
  print(f"Output: {output.outputs[0].text}")

