from SLM.slm import SLM
import configuration
import csv
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

  def generate_multiple_feedback(self, output_filename: str | None = None) -> list[dict[str, object]]:
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

    if output_filename:
      self._save_feedback_to_csv(all_feedback, output_filename)

    return all_feedback

  def _save_feedback_to_csv(self, all_feedback: list[dict[str, object]], output_filename: str) -> Path:
    output_path = Path(output_filename)

    if output_path.name != output_filename:
      raise ValueError("output_filename must be a filename only, not a path.")

    if output_path.suffix != ".csv":
      output_path = output_path.with_suffix(".csv")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / output_path

    rows = []

    for essay_feedback in all_feedback:
      essay_file = essay_feedback["essay_file"]
      responses = essay_feedback["responses"]

      for response_index, response in enumerate(responses, start=1):
        rows.append({
          "essay_file": essay_file,
          "response_index": response_index,
          "response": response
        })

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
      writer = csv.DictWriter(
        csv_file,
        fieldnames=[
          "essay_file",
          "response_index",
          "response"
        ]
      )
      writer.writeheader()
      writer.writerows(rows)

    return output_path
