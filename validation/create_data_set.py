# Loop through BAWE files and select paragraphs with 8 sentences.
# We want 400 8-sentence paragraphs.
# Each paragraph should start at a paragraph start.
# Save them to texts/originals
# Inside texts keep a csv file called metadata.csv that tracks the texts with the following information
# filename, word_count, domain, subject, type
# We want 100 from each of AH, LS, PS and SS
# AH has 8 subjects, so select around 12 from each subject randomly
# LS has 6 subjects, so select around 20 from each subject randomly
# PS has 10 subjects, so select around 10 from each subject randomly
# SS has 9 subjects, so select around 11 from each subject randomly

from dataclasses import dataclass
from pathlib import Path
import random
import csv
import re

@dataclass(frozen=True)
class CorpusFile:
  domain: str
  subject: str
  filename: str
  path: Path


class CreateDataSet:

  def __init__(self) -> None:
    self.corpus_data: list[CorpusFile] = []
    self.sampled_corpus: list[CorpusFile] = []

  def scan_for_files(self, corpus_path: str) -> None:
    """
    Scans all the files in /Volumes/BAWE_TXT and creates a data type that stores the following information
    Domain | Subject | Filename
    AH | Archaeology | AH_Archaeology_Critique_1_6029c.txt
    ...

    This is saved to corpus_data variable
    """
    corpus_root = Path(corpus_path)

    if not corpus_root.is_dir():
      raise FileNotFoundError(
        f"Corpus directory does not exist: {corpus_root}"
      )

    records: list[CorpusFile] = []

    for file_path in sorted(corpus_root.rglob("*.txt")):
      if not file_path.is_file():
        continue
      relative_parts = file_path.relative_to(corpus_root).parts

      if len(relative_parts) < 3:
        raise ValueError(
          f"Expected domain/subject/filename structure, but found {file_path}"
        )

      domain = relative_parts[0]
      subject = relative_parts[1]

      records.append(
        CorpusFile(
          domain=domain,
          subject=subject,
          filename=file_path.name,
          path=file_path
        )
      )
    

    self.corpus_data = records
    print(self.corpus_data)

  def sample_corpus_texts(
      self,
      samples_per_domain: int = 100,
      seed: int | None = None
      ) -> None:
    """
    Selects a representative sample of texts, 100 from each Domain
    with an approximately equal amount randomly selected from each Subject within each Domain
    Saved to sampled_corpus variable which has the same shape as corpus_data
    """
    rng = random.Random(seed)

    selected_files: list[CorpusFile] = []

    for domain in ("AH", "LS", "PS", "SS"):
      domain_files = [
        corpus_file
        for corpus_file in self.corpus_data
        if corpus_file.domain == domain
      ]

      domain_sample = rng.sample(
        domain_files,
        samples_per_domain
      )

      selected_files.extend(domain_sample)

    self.sampled_corpus = selected_files

    self.sampled_corpus
    return None

  def extract_text(
      self,
      output_dir: str | Path = "texts/originals",
      metadata_path: str | Path = "texts/metadata.csv",
      seed: int | None = None
  ) -> None:
    """
    Loops through each self.sampled_corpus text file
    Extracts a random paragraph. Ensures it has at least 8 sentences.
    Copies the first 8 sentences from that paragraph to a new file.
    New file is saved in texts/originals using the same file name as where it came from, but the folder structure is flattened (no Domain or Subject folders)
    Updates the metadata.csv file in texts folder with the data about the file, including word count
    """
    if not self.sampled_corpus:
      raise ValueError(
        "sampled_corpus is empty. Run sample_corpus_texts first."
      )

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for corpus_file in self.sampled_corpus:
      source_text = corpus_file.path.read_text(encoding="utf-8")
      # Paragraphs are separated by one or more blank lines.

      paragraphs = re.split(
        r"\n\s*\n",
        source_text.strip(),
      )

      eligible_paragraphs: list[list[str]] = []

      for paragraph in paragraphs:
        # Convert internal newlines and repeated spaces to single psaces.
        paragraph = re.sub(
          r"\s+", " ", paragraph.strip()
        )

        if not paragraph:
          continue

        # Split on sentence-ending punctuation followed by whitespace.
        sentences = [
          sentence.strip()
          for sentence in re.split(
            r"(?<=[.!?])\s+",
            paragraph
          )
          if sentence.strip()
        ]

        if len(sentences) >= 8:
          eligible_paragraphs.append(sentences)

      selected_paragraph =rng.choice(eligible_paragraphs)
      extracted_text = " ".join(selected_paragraph[:8]) + "\n"

      output_path = output_directory / corpus_file.filename
      output_path.write_text(
        extracted_text,
        encoding="utf-8"
      )

      self.update_metadata(
        corpus_file=corpus_file,
        output_path=output_path,
        extracted_text=extracted_text,
        metadata_path=metadata_path
      )
    return None

  def word_count(self, text: str) -> int:
    words = re.findall(
      r"\b[\w]+(?:[-'][\w]+)*\b",
      text,
    )
    return len(words)

  def update_metadata(
      self,
      corpus_file: CorpusFile,
      output_path: Path,
      extracted_text: str,
      metadata_path: str | Path = "texts/metadata.csv"
  ) -> None:
    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
      "filename",
      "word_count",
      "domain",
      "subject",
      "type",
    ]

    file_is_empty_or_missing = (
      not metadata_file.is_file()
      or metadata_file.stat().st_size == 0
    )

    with metadata_file.open(
      "a",
      newline="",
      encoding="utf-8"
    ) as file:
      writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
      )

      if file_is_empty_or_missing:
        writer.writeheader()

      writer.writerow(
        {
          "filename": output_path.name,
          "word_count": self.word_count(extracted_text),
          "domain": corpus_file.domain,
          "subject": corpus_file.subject,
          "type": "original",
        }
      )
  