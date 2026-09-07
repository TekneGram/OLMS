from llama_cpp import Llama
import configuration
import json
import re

CONTENT = (
  "You are an essay feedback tutor."
)

messages = [
  {
    "role": "system", "content": CONTENT
  }
]

class SLM:
  def __init__(
      self
  ) -> None:
    self.llm = self._getAI()
    return

  def _getAI(self) -> Llama:
    llm = Llama(
      model_path=configuration.llm_path,
      n_ctx=4096,
      n_threads=8,
      n_gpu_layers=999,
      verbose=False
    )
    return llm

  def run_inference(self, user_request) -> str:
    response = self.llm.create_chat_completion(
      messages=user_request,
      max_tokens=120,
      temperature=0.7,
      top_p=0.95,
      stream=False
    )

    llm_message = response["choices"][0]["message"]
    tutor_message = (llm_message.get("content") or "").strip()

    return tutor_message