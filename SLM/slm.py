from llama_cpp import Llama
import configuration
import re

CONTENT = (
  "You are an AI tutor. "
  "Do not include hidden reasoning, chain-of-thought, or <think> tags."
)

SYSTEM_MESSAGES = [
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
      n_ctx=configuration.n_ctx,
      n_threads=configuration.n_threads,
      n_gpu_layers=configuration.n_gpu_layers,
      verbose=configuration.verbose
    )
    return llm

  def run_inference(self, user_request) -> str:
    chat_messages = [
      *SYSTEM_MESSAGES,
      {
        "role": "user",
        "content": f"{user_request}\n\n/no_think"
      }
    ]

    response = self.llm.create_chat_completion(
      messages=chat_messages,
      max_tokens=configuration.max_tokens,
      temperature=configuration.temperature,
      top_p=configuration.top_p,
      stream=configuration.stream
    )

    llm_message = response["choices"][0]["message"]
    tutor_message = (llm_message.get("content") or "").strip()
    tutor_message = re.sub(r"<think>.*?</think>", "", tutor_message, flags=re.DOTALL).strip()
    tutor_message = re.sub(r"</?think>", "", tutor_message).strip()

    return tutor_message
