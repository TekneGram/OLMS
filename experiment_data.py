"""CSV and metadata helpers shared by collection and vector generation."""

import csv
import hashlib
import json
import os
from pathlib import Path


RESPONSE_COLUMNS = ["essay_file", "prompt", "response_index", "response"]


def file_digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_path(path):
  return Path(path).with_suffix(".metadata.json")


def write_metadata(path, metadata):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
  temporary.replace(path)


def append_csv_row(handle, writer, row):
  writer.writerow(row)
  handle.flush()
  os.fsync(handle.fileno())


def read_responses(path, complete=True, expected_count=None):
  with Path(path).open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames != RESPONSE_COLUMNS:
      raise ValueError("Response CSV must have essay_file,prompt,response_index,response columns. Legacy feedback needs regeneration with the responses command.")
    rows = list(reader)

  seen, groups = set(), {}

  for row in rows:
    if set(row) != set(RESPONSE_COLUMNS) or any(value is None for value in row.values()):
      raise ValueError("Malformed or interrupted response CSV row.")
    if not row["essay_file"].strip() or row["prompt"] not in ["A", "B"] or not row["response"].strip():
      raise ValueError("Responses require an essay, an A/B prompt label, and nonempty text.")
    try:
      index = int(row["response_index"])
    except (ValueError, TypeError) as error:
      raise ValueError("Response indices must be positive integers.") from error
    if index < 1 or (expected_count is not None and index > expected_count):
      raise ValueError("Response index is outside the requested repetition count.")
    row["response_index"] = index
    key = (row["essay_file"], row["prompt"], index)
    if key in seen:
      raise ValueError(f"Duplicate response identifier: {key}")
    seen.add(key)
    groups.setdefault(key[:2], set()).add(index)

  if complete:
    if not rows:
      raise ValueError("Response CSV is empty.")
    count = expected_count if expected_count is not None else max(row["response_index"] for row in rows)
    if count < 2:
      raise ValueError("At least two responses per essay and prompt are required.")
    for essay in {row["essay_file"] for row in rows}:
      for prompt in ["A", "B"]:
          if groups.get((essay, prompt)) != set(range(1, count + 1)):
            raise ValueError(f"Incomplete responses for {essay}, prompt {prompt}; expected indices 1..{count}.")

  return rows
