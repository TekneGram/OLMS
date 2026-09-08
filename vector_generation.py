"""Generate and checkpoint OLMS pairs from an already collected response CSV."""

import csv
import json
from itertools import combinations_with_replacement, product
from pathlib import Path
from collections.abc import Callable
from typing import Any

ResponseRow = dict[str, Any]
VectorScores = dict[str, float]

import pandas as pd

from analysis.olms_vector_table import OLMSVectorTable
from experiment_data import append_csv_row, file_digest, metadata_path, read_responses, write_metadata


def generate_vectors(
        responses_path: str | Path,
        output_path : str | Path,
        *,
        resume: bool = False,
        parser_dir: str | Path = ".models/udpipe",
        diagnostics: bool = False,
        score_pair: Callable[[ResponseRow, ResponseRow], VectorScores] | None = None
    ):

  # Convert input/output paths to Path objects
  source, output = Path(responses_path), Path(output_path)
  if source.resolve() == output.resolve():
    raise ValueError("Vector output must differ from response input.")

  response_metadata = None
  # Load response metadata and require collection to be complete before scoring.
  if metadata_path(source).exists():
    response_metadata = json.loads(metadata_path(source).read_text(encoding="utf-8"))
    if not response_metadata.get("complete"):
      raise ValueError("Response collection is incomplete. Resume the responses command first.")

  # Validate saved responses and discover the essays represented in the file.
  responses = read_responses(source, expected_count=response_metadata["repeats"] if response_metadata else None)

  essays = sorted({r["essay_file"] for r in responses})

  if response_metadata and set(essays) != set(response_metadata["essay_hashes"]):
    raise ValueError("Response CSV does not contain all essays recorded in its metadata.")

  # Determine the per-prompt response count used to build pair combinations.
  count = max(r["response_index"] for r in responses)
  root = Path(__file__).resolve().parent
  algorithm_files = [Path(__file__), root / "scoring.py",
                      *sorted((root / "OLMSClasses").glob("*.py")),
                      *sorted((root / "TextProcessing").glob("*.py"))]

  # Fingerprint inputs, scorer code, and settings so resume cannot mix experiments.
  metadata = {"schema_version": 1, "response_sha256": file_digest(source),
              "response_file": str(source.resolve()), "essay_files": essays, "response_count": count,
              "algorithm_sha256": {str(p.relative_to(root)): file_digest(p) for p in algorithm_files},
              "parser_dir": str(Path(parser_dir).resolve()), "diagnostics": diagnostics,
              "response_metadata": response_metadata, "complete": False}

  completed = set()

  # When resuming, load already completed pairs after checking compatibility.
  if output.exists():
      if not resume:
        raise FileExistsError(f"Vectors already exist: {output}. Use --resume or a new --output path.")
      previous = json.loads(metadata_path(output).read_text(encoding="utf-8"))
      if any(previous.get(key) != value for key, value in metadata.items() if key != "complete"):
        raise ValueError("Response data or vector settings changed. Use a new vector output file.")
      frame = pd.read_csv(output, keep_default_na=False)
      OLMSVectorTable.validate_rows(frame)
      if not frame.essay_file.isin(essays).all() or not (frame.response_count == count).all():
        raise ValueError("Checkpoint vectors do not match the response dataset.")
      completed = set(frame[OLMSVectorTable.KEYS].itertuples(index=False, name=None))
  elif resume:
      raise FileNotFoundError(f"Cannot resume missing vectors: {output}")

  output.parent.mkdir(parents=True, exist_ok=True)
  write_metadata(metadata_path(output), metadata)

  diagnostics_dir = output.parent / f"{output.stem}_diagnostics" if diagnostics else None
  if diagnostics_dir is not None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

  # Index responses by essay, prompt, and repetition for pairwise lookup.
  lookup = {(r["essay_file"], r["prompt"], r["response_index"]): r for r in responses}
  fresh = not output.exists()
  total = len(essays) * (count * (count + 1) + count * count)

  with output.open("x" if fresh else "a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=OLMSVectorTable.COLUMNS)
    if fresh:
      writer.writeheader()
      handle.flush()

    # Score all AA, BB, and AB response pairs for each essay, skipping checkpointed rows.
    for essay in essays:
      for p, q in [("A", "A"), ("B", "B"), ("A", "B")]:
        ids = range(1, count + 1)
        pairs = product(ids, ids) if p != q else combinations_with_replacement(ids, 2)
        for i, j in pairs:
          key = (essay, p, i, q, j)
          if key in completed:
            continue
          # Build the real OLMS scorer only when there is work to do.
          if score_pair is None:
            # lazy importing
            from scoring import OLMSPairScorer
            score_pair = OLMSPairScorer(parser_dir, diagnostics_dir)

          # Compute one OLMS vector and immediately checkpoint it to CSV.
          vector = score_pair(lookup[(essay, p, i)], lookup[(essay, q, j)])
          row = {**dict(zip(OLMSVectorTable.KEYS, key)), "comparison_type": p + q,
                  "response_count": count, **vector}
          OLMSVectorTable.validate_rows(pd.DataFrame([row]))
          append_csv_row(handle, writer, row)
          completed.add(key)
          print(f"Saved vector {len(completed)}/{total}: {essay} {p}{i}/{q}{j}", flush=True)

  # Revalidate the finished table before marking vector metadata complete.
  OLMSVectorTable(output)
  metadata["complete"] = True
  write_metadata(metadata_path(output), metadata)
  return output
