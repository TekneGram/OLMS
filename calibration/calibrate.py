"""Generate OLMS calibration vectors for sections I, II, and III."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable


SECTION_PATTERN = re.compile(r"(?m)^##\s+(I|II|III|IV)\s*$")
COMPARISONS = (
  ("I", "II"),
  ("I", "III"),
  ("II", "III"),
  ("I", "IV"),
  ("II", "IV"),
  ("III", "IV"),
)
COMPONENTS = ("org", "lex", "meaning", "struct")


def extract_sections(text: str) -> dict[str, str]:
  """Return the paragraph text under the exact I, II, and III headings."""
  matches = list(SECTION_PATTERN.finditer(text))
  sections: dict[str, str] = {}

  for index, match in enumerate(matches):
    section = match.group(1)
    if section in sections:
      raise ValueError(f"Duplicate section heading: ## {section}")

    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    sections[section] = text[match.end():end].strip()

  section_order = ("I", "II", "III", "IV")
  missing = [section for section in section_order if section not in sections]
  if missing:
    raise ValueError(f"Missing calibration section(s): {', '.join(missing)}")

  empty = [section for section, content in sections.items() if not content]
  if empty:
    raise ValueError(f"Empty calibration section(s): {', '.join(empty)}")

  return {section: sections[section] for section in section_order}


def _response_record(filename: str, section: str, response: str, index: int) -> dict:
  return {
    "essay_file": filename,
    "prompt": section,
    "response_index": index,
    "response": response,
  }


def run_calibration(
    texts_dir: str | Path = Path("calibration/texts"),
    output_path: str | Path = Path("calibration/calibration_results.md"),
    parser_dir: str | Path = Path(".models/udpipe"),
    scorer_factory: Callable | None = None,
) -> Path:
  texts_dir = Path(texts_dir)
  output_path = Path(output_path)
  text_paths = sorted(texts_dir.glob("*.md"))

  if not texts_dir.is_dir():
    raise ValueError(f"Calibration text directory does not exist: {texts_dir}")
  if not text_paths:
    raise ValueError(f"No Markdown calibration texts found in: {texts_dir}")

  if scorer_factory is None:
    from scoring import OLMSPairScorer
    scorer_factory = OLMSPairScorer

  results: list[dict] = []
  scorer = scorer_factory(parser_dir)
  try:
    for text_path in text_paths:
      sections = extract_sections(text_path.read_text(encoding="utf-8"))
      records = {
        section: _response_record(text_path.name, section, sections[section], index)
        for index, section in enumerate(("I", "II", "III", "IV"), start=1)
      }

      for first_section, second_section in COMPARISONS:
        scores = scorer.score_pair(records[first_section], records[second_section])
        results.append({
          "file": text_path.name,
          "comparison": f"{first_section}-{second_section}",
          "scores": scores,
        })
  finally:
    scorer.close()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(_render_report(results), encoding="utf-8")
  return output_path


def _render_report(results: list[dict]) -> str:
  lines = ["# OLMS Calibration Results", ""]
  current_file = None

  for result in results:
    if result["file"] != current_file:
      if current_file is not None:
        lines.append("")
      current_file = result["file"]
      lines.extend([
        f"## {current_file}",
        "",
        "| Comparison | Organization | Lexical | Meaning | Structural |",
        "|---|---:|---:|---:|---:|",
      ])

    scores = result["scores"]
    values = " | ".join(f"{float(scores[component]):.6f}" for component in COMPONENTS)
    lines.append(f"| {result['comparison']} | {values} |")

  return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Generate OLMS calibration results.")
  parser.add_argument("--texts", type=Path, default=Path("calibration/texts"))
  parser.add_argument("--output", type=Path, default=Path("calibration/calibration_results.md"))
  parser.add_argument("--parser-dir", type=Path, default=Path(".models/udpipe"))
  return parser


def main() -> None:
  args = build_parser().parse_args()
  output = run_calibration(args.texts, args.output, args.parser_dir)
  print(f"Saved calibration results to {output}")


if __name__ == "__main__":
  main()
