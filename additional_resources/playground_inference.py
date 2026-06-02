"""
Launch OpenAI compatible server using vLLM

> pip install vllm[audio]

> vllm serve google/gemma-4-E4B-it \
  --port 8001 \
  --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt '{"image": 1, "audio": 1}'
"""
import base64
from textwrap import dedent
from typing import Any

from openai import OpenAI

client = OpenAI(
    base_url="http://13.223.245.59:8001/v1",
    api_key="EMPTY"
)

system_prompt = dedent("Assume you are a Math and physics professor teaching for graduation and higher students at universities. "
       "Keeping your role in mind without hallucination answer clearly and very focused to the user questions")

user_prompt = dedent("Explain What is hessian Matrix and where is it used in Machine Learning?")

response = client.chat.completions.create(
    model="google/gemma-4-E4B-it",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=512,
    temperature=0.3
)

print(response.choices[0].message.content)

########### Audio Transcription ############

# def to_data_url(path: str) -> str:
#     with open(path, "rb") as fh:
#         return "data:audio/mpeg;base64," + base64.b64encode(fh.read()).decode("utf-8")
#
# transcription: Any = client.chat.completions.create(
#     model="google/gemma-4-E4B-it",
#     messages=[
#         {"role": "user", "content": [
#             {
#                 "type": "audio_url",
#                 "audio_url": {"url": to_data_url("gemma4.mp3")}
#             },
#             {
#                 "type": "text",
#                 "text": "Provide a verbatim, word-for-word transcription of the audio."
#             }
#         ]}
#     ],
#     max_tokens=512,
#     temperature=0.3
# )

print(transcription)