from llama_cpp import Llama
import configuration
import re
import json

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
      verbose=configuration.verbose,
      seed=configuration.seed,
    )
    return llm

  def run_inference(self, user_request, seed=None) -> str:
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
      stream=configuration.stream,
      seed=seed,
      #response_format=configuration.ai_response_format
    )

    tutor_message = ""

    if configuration.stream:
      for chunk in response:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")

        if content:
          print(content, end="", flush=True)
          tutor_message += content
    else:
      tutor_message = response["choices"][0]["message"].get("content") or ""

    print()

    # llm_message = response["choices"][0]["message"]
    # tutor_message = (llm_message.get("content") or "").strip()
    tutor_message = tutor_message.strip()
    tutor_message = re.sub(r"<think>.*?</think>", "", tutor_message, flags=re.DOTALL).strip()
    tutor_message = re.sub(r"</?think>", "", tutor_message).strip()

    try:
      parsed_response = json.loads(tutor_message)
      is_json=True
    except json.JSONDecodeError:
      parsed_response = tutor_message
      is_json=False

    if is_json:

      feedback_json = json.loads(tutor_message)

      required_fields = configuration.ai_response_required_fields

      missing_fields = [
        field
        for field in required_fields
        if field not in feedback_json
      ]

      if missing_fields:
        raise ValueError(
          f"LLM response is missing required JSON fields: {missing_fields}"
        )

      for field in required_fields:
        if not isinstance(feedback_json[field], str):
          raise ValueError(f"LLM JSON field must be a string: { field }")

      tutor_message = "\n\n".join(
        feedback_json[field].strip()
        for field in required_fields
        if feedback_json[field].strip()
      )

    else:
      tutor_message = parsed_response.strip()

    return tutor_message

  def set_system_content(self, system_content) -> None:
    self.system_content = system_content
