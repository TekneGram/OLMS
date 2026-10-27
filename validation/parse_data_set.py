
import csv
import re
from pathlib import Path

class ParseDataSet:

  def __init__(self) -> None:
    self.originals_dir = Path("texts/originals")
    self.parsed_dir = Path("texts/parsed")
    self.metadata_path = Path("texts/metadata.csv")
    return None

  def split_sentences(self, text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [
      sentence.strip()
      for sentence in re.split(
        r"(?<=[.!?])\s+",
        text,
      )
      if sentence.strip()
    ]

  def parse(self) -> None:
    """
    Loop through all the files in texts/originals folder
    Read one text at a time.
    Loop through four conditions: same, intro_conc_swap, adjacent_swap, reverse
    For same, we create a copy of the text that is identical. It is saved into a same-named file in texts/parsed with "I \n" + text + "\n II \n" + same text
    For intro_conc_swap, create new text with first and second sentences swapped. Save into same-named file in texts/parsed with "I \n" + text + "\n II \n" + intro_conc_swap text
    For adjacent_swap, create a new text with sentences 1-2, 3-4, 5-6, 7-8 all swapped so that the order is now 2-1, 4-3, 6-5, 8-7. Save into same-named file in texts/parsed with "I \n" + text + "\n II \n" + adjacent_swap text
    For reverse, create a new text with sentences in reverse order, 8-7-6-5-4-3-2-1. Save into same-named file in texts/parsed with "I \n" + text + "\n II \n" + reversed text
    For each new text, update the texts/metadata.csv file to add a column "edit" which is either "same", "intro_conc_swap", "adjacent_swap" or "reverse"
    Use the functions keep_same, intro_conc_swap, adjacent_swap, reverse to make the text changes.
    """
    with self.metadata_path.open(
      "r",
      newline="",
      encoding="utf-8",
    ) as file:
      metadata_rows = list(csv.DictReader(file))

    metadata_by_filename = {
      row["filename"]: row
      for row in metadata_rows
    }

    conditions = [
      ("same", self.keep_same),
      ("intro_conc_swap", self.intro_conc_swap),
      ("adajcent_swap", self.adjacent_swap),
      ("reverse", self.reverse)
    ]

    parsed_metadata: list[dict[str, str]] = []

    original_files = sorted(
      self.originals_dir.glob("*.txt")
    )

    for file_number, original_path in enumerate(original_files):
      original_text = original_path.read_text(
        encoding="utf-8"
      ).strip()

      condition_name, transform = conditions[
        file_number % len(conditions)
      ]

      transformed_text = transform(original_text)

      parsed_text = (
        f"I\n"
        f"{original_text}\n"
        f"II\n"
        f"{transformed_text}\n"
      )

      self.parsed_dir.mkdir(
        parents=True,
        exist_ok=True
      )

      output_path = self.parsed_dir / original_path.name
      output_path.write_text(
        parsed_text,
        encoding="utf-8",
      )

      if original_path.name not in metadata_by_filename:
        raise ValueError(
          f"No metadata found for {original_path.name}"
        )

      metadata_row = metadata_by_filename[
        original_path.name
      ].copy()

      metadata_row["edit"] = condition_name
      parsed_metadata.append(metadata_row)

    fieldnames = [
      "filename",
      "word_count",
      "domain",
      "subject",
      "type",
      "edit"
    ]

    with self.metadata_path.open(
      "w",
      newline="",
      encoding="utf-8",
    ) as file:
      writer=csv.DictWriter(
        file,
        fieldnames=fieldnames
      )

      writer.writeheader()
      writer.writerows(parsed_metadata)


  def keep_same(self, text: str) -> str:
    return text

  def intro_conc_swap(self, text: str) -> str:
    """
    Create new_text such that the 1st and 8th sentences are swapped
    """
    sentences = self.split_sentences(text)
    sentences[0], sentences[7] = (
      sentences[7], sentences[0]
    )

    return " ".join(sentences)

  def adjacent_swap(self, text: str) -> str:
    """
    Create new text such that sentences 1 and 2, 3 and 4, 5 and 6, 7 and 8 are swapped
    to create 2-1-4-3-6-5-8-7
    """
    sentences = self.split_sentences(text)
    swapped: list[str] = []

    for index in range(0, 8, 2):
      swapped.extend(
        [
          sentences[index + 1],
          sentences[index],
        ]
      )
    return " ".join(swapped)

  def reverse(self, text: str) -> str:
    """
    Create new text such that all sentences are reversed
    to create 8-7-6-5-4-3-2-1
    """
    sentences = self.split_sentences(text)
    return " ".join(reversed(sentences))