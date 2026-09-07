from llama_cpp import Llama
import configuration
import re

CONTENT = (
  "You are an AI tutor. "
  "Do not include hidden reasoning, chain-of-thought, or <think> tags."
)

class SLM:
  def __init__(
      self
  ) -> None:
    self.llm = self._getAI()
    self.system_content = CONTENT
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
    system_messages = [
      {
        "role": "system",
        "content": self.system_content,
      }
    ]
    chat_messages = [
      *system_messages,
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

  def set_system_content(self, system_content) -> None:
    self.system_content = system_content
