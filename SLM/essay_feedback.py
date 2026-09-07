from SLM.slm import SLM
import configuration
from pathlib import Path

class EssayFeedback:
  def __init__(self) -> None:
    self.llm = SLM()
    return

  def generate_essay_feedback(self, prompts: list[str], system_content: str | None = None) -> list[str]:

    if system_content:
      self.llm.set_system_content(system_content)

    responses = []
    for prompt in prompts:
      responses.append(self.llm.run_inference(prompt))

    return responses

  def generate_multiple_feedback(self) -> list[dict[str, object]]:
    # Loop through all the files in the essay path provided by configuration
    essays_path = Path(configuration.essays_path)

    if not essays_path.exists():
      raise FileNotFoundError(f"Essay folder does not exist: {essays_path}")

    if not essays_path.is_dir():
      raise NotADirectoryError(f"Essay path is not a folder: {essays_path}")

    system_content = (
      configuration.ai_role
      + configuration.ai_task_main
      + "".join(configuration.ai_knowledge)
    )

    all_feedback = []

    for essay_file in essays_path.glob("*.md"):
      essay = essay_file.read_text(encoding="utf-8").strip()

      prompt_1 = configuration.ai_task_description + essay + "\n\n" + configuration.ai_recipient_information_1
      prompt_2 = configuration.ai_task_description + essay + "\n\n" + configuration.ai_recipient_information_2
      prompts = [prompt_1, prompt_2]
      responses = self.generate_essay_feedback(prompts, system_content)

      all_feedback.append({
        "essay_file": essay_file.name,
        "responses": responses
      })

    return all_feedback
