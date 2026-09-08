import csv
import hashlib
import json
from pathlib import Path
from SLM.slm import SLM
from typing import Any

import configuration
from experiment_data import (
  RESPONSE_COLUMNS, append_csv_row, metadata_path, read_responses, write_metadata,
)


class EssayFeedback:
  """Collect repeated feedback and checkpoint each response before continuing."""

  def __init__(self, llm=None) -> None:
    self.llm = llm


  def _model(self) -> SLM:
    if self.llm is None:
      self.llm = SLM()
    return self.llm


  def generate_essay_feedback(
        self, 
        prompts: list[str], 
        system_content: str | None = None
    ) -> list[str]:
      model = self._model()
      if system_content:
        model.set_system_content(system_content)
      return [model.run_inference(prompt) for prompt in prompts]

  # Loads essays
  # Validates / resume-checks metadata
  # Writes metadata with complete: False
  # Builds system report
  # Selects essay + prompt + repetition index
  # skips row if it already exists during --resume
  # set the model's system content
  # computes the deterministic seed
  # Hashes ensure that any changes to essays will start a new experiment so that experiments don't get muddled up.
  def generate_multiple_feedback(
          self, 
          output_filename: str | None = None, 
          repeats: int =10, 
          *,
          output_path: Path | str | None = None,
          essays_path: Path | str | None = None,
          resume: bool = False
    ) -> list[dict[str, Any]]:
      if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats must be an integer of at least two.")

      folder = Path(essays_path or configuration.essays_path)
      if not folder.is_dir():
        raise NotADirectoryError(f"Essay folder does not exist: {folder}")

      essays = {p.name: p.read_text(encoding="utf-8").strip() for p in sorted(folder.glob("*.md"))}
      if not essays or any(not text for text in essays.values()):
        raise ValueError("Essay folder must contain nonempty .md essays.")

      if output_path is None:
        name = Path(output_filename or "feedback_data")
        if name.name != str(name):
          raise ValueError("output_filename must be a filename; use output_path for paths.")
        output_path = Path("data") / name.with_suffix(".csv")

      path = Path(output_path)
      path.parent.mkdir(parents=True, exist_ok=True)
      settings = {name: getattr(configuration, name) for name in [
        "llm_path", "n_ctx", "n_threads", "n_gpu_layers", "seed", "max_tokens",
        "temperature", "top_p", "stream", "ai_role", "ai_task_main", "ai_knowledge",
        "ai_task_description", "ai_recipient_information_1", "ai_recipient_information_2",
      ]}
      settings = json.loads(json.dumps(settings))
      hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in essays.items()}
      metadata = {"schema_version": 1, "repeats": repeats, "settings": settings,
                  "essay_hashes": hashes, "complete": False,
                  "seed_policy": "first 32 bits of sha256(JSON [base_seed, essay_filename, prompt, repetition])"}

      existing = []
      if path.exists():
        if not resume:
          raise FileExistsError(f"Responses already exist: {path}. Use --resume or a new --output path.")

        previous = json.loads(metadata_path(path).read_text(encoding="utf-8"))

        if any(previous.get(key) != metadata[key] for key in ["schema_version", "repeats", "settings", "seed_policy"]):
          raise ValueError("Response settings changed; use a new output file for a new experiment.")

        if any(hashes.get(name) != digest for name, digest in previous["essay_hashes"].items()):
          raise ValueError("Previously collected essays were changed or removed. Use a new output file.")

        existing = read_responses(path, complete=False, expected_count=repeats)

        if any(row["essay_file"] not in previous["essay_hashes"] for row in existing):
          raise ValueError("Response CSV contains essays absent from its metadata.")

      elif resume:
        raise FileNotFoundError(f"Cannot resume missing responses: {path}")

      completed = {(r["essay_file"], r["prompt"], r["response_index"]) for r in existing}
      write_metadata(metadata_path(path), metadata)

      system = configuration.ai_role + configuration.ai_task_main + "".join(configuration.ai_knowledge)
      recipients = {"A": configuration.ai_recipient_information_1, "B": configuration.ai_recipient_information_2}
      fresh = not path.exists()

      with path.open("x" if fresh else "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        if fresh:
          writer.writeheader()
          handle.flush()
        for essay_name, essay in essays.items():
          for prompt, recipient in recipients.items():
            request = configuration.ai_task_description + essay + "\n\n" + recipient
            for index in range(1, repeats + 1):
              if (essay_name, prompt, index) in completed:
                continue

              # Get the model
              model = self._model()
              model.set_system_content(system)
              seed_text = json.dumps([configuration.seed, essay_name, prompt, index])
              seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "big")

              # Run inference!
              response = model.run_inference(request, seed=seed)

              if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Empty response for {essay_name}, {prompt}, {index}.")

              row = {"essay_file": essay_name, "prompt": prompt,
                      "response_index": index, "response": response.strip()}
              append_csv_row(handle, writer, row)
              existing.append(row)
              print(f"Saved {essay_name}: prompt {prompt}, response {index}/{repeats}", flush=True)

      metadata["complete"] = True
      write_metadata(metadata_path(path), metadata)
      return existing
