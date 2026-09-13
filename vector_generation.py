"""Generate and checkpoint OLMS pairs from an already collected response CSV."""

import csv
import json
import multiprocessing
import queue
import traceback
from itertools import combinations_with_replacement, product
from pathlib import Path
from collections.abc import Callable
from typing import Any

ResponseRow = dict[str, Any]
VectorScores = dict[str, float]
VectorKey = tuple[str, str, int, str, int]
PairWork = tuple[VectorKey, ResponseRow, ResponseRow]

import pandas as pd

import configuration
from analysis.olms_vector_table import OLMSVectorTable
from experiment_data import append_csv_row, file_digest, metadata_path, read_responses, write_metadata


def _parallel_score_worker(
    parser_dir: str | Path,
    task_queue: Any,
    result_queue: Any,
) -> None:
  """Score complete essays in an isolated process with one reusable model set."""
  scorer = None
  active_essay = None
  try:
    from scoring import OLMSPairScorer

    scorer = OLMSPairScorer(parser_dir)
    while (task := task_queue.get()) is not None:
      active_essay, pairs = task
      scorer.prepare_embeddings([
        response for _, first, second in pairs for response in (first, second)
      ])
      scores = scorer.score_pairs([(first, second) for _, first, second in pairs])
      for (key, _, _), score in zip(pairs, scores, strict=True):
        result_queue.put(("row", key, score))
      result_queue.put(("complete", active_essay))
      active_essay = None
  except BaseException:
    result_queue.put(("error", active_essay, traceback.format_exc()))
  finally:
    if scorer is not None:
      scorer.close()


def _append_vector_row(
    handle: Any,
    writer: csv.DictWriter,
    key: VectorKey,
    vector: VectorScores,
    count: int,
    completed: set[VectorKey],
    total: int,
) -> None:
  """Validate and checkpoint one vector row in canonical order."""
  _, p, i, q, j = key
  row = {**dict(zip(OLMSVectorTable.KEYS, key)), "comparison_type": p + q,
          "response_count": count, **vector}
  OLMSVectorTable.validate_rows(pd.DataFrame([row]))
  append_csv_row(handle, writer, row)
  completed.add(key)
  print(f"Saved vector {len(completed)}/{total}: {key[0]} {p}{i}/{q}{j}", flush=True)


def _score_parallel(
    work_by_essay: list[tuple[str, list[PairWork]]],
    parser_dir: str | Path,
    workers: int,
    handle: Any,
    writer: csv.DictWriter,
    count: int,
    completed: set[VectorKey],
    total: int,
) -> None:
  """Score essays concurrently while the parent preserves checkpoint order."""
  context = multiprocessing.get_context("spawn")
  task_queue = context.Queue()
  result_queue = context.Queue()
  worker_count = min(workers, len(work_by_essay))
  processes = [
    context.Process(target=_parallel_score_worker, args=(parser_dir, task_queue, result_queue))
    for _ in range(worker_count)
  ]
  expected_keys = [key for _, pairs in work_by_essay for key, _, _ in pairs]
  buffered: dict[VectorKey, VectorScores] = {}
  next_key = 0
  next_task = 0
  active_tasks = 0

  def submit_next_task() -> bool:
    nonlocal next_task, active_tasks
    if next_task == len(work_by_essay):
      return False
    task_queue.put(work_by_essay[next_task])
    next_task += 1
    active_tasks += 1
    return True

  try:
    for process in processes:
      process.start()
    for _ in processes:
      submit_next_task()

    while active_tasks:
      try:
        message = result_queue.get(timeout=0.5)
      except queue.Empty:
        failed = next((process for process in processes if process.exitcode not in (None, 0)), None)
        if failed is not None:
          raise RuntimeError(f"Vector worker exited unexpectedly with code {failed.exitcode}.")
        continue

      event = message[0]
      if event == "row":
        _, key, vector = message
        buffered[key] = vector
        while next_key < len(expected_keys) and expected_keys[next_key] in buffered:
          ordered_key = expected_keys[next_key]
          _append_vector_row(
            handle, writer, ordered_key, buffered.pop(ordered_key), count, completed, total)
          next_key += 1
      elif event == "complete":
        active_tasks -= 1
        submit_next_task()
      elif event == "error":
        _, essay, details = message
        raise RuntimeError(f"Vector worker failed while scoring {essay}:\n{details}")
      else:
        raise RuntimeError(f"Vector worker returned an unknown event: {event!r}")

    if next_key != len(expected_keys):
      raise RuntimeError("Parallel vector workers did not return every expected vector row.")
    for _ in processes:
      task_queue.put(None)
    for process in processes:
      process.join()
      if process.exitcode != 0:
        raise RuntimeError(f"Vector worker exited unexpectedly with code {process.exitcode}.")
  finally:
    for process in processes:
      if process.is_alive():
        process.terminate()
      process.join()
    task_queue.close()
    result_queue.close()


def generate_vectors(
        responses_path: str | Path,
        output_path : str | Path,
        *,
        resume: bool = False,
        parser_dir: str | Path = ".models/udpipe",
        diagnostics: bool = False,
        score_pair: Callable[[ResponseRow, ResponseRow], VectorScores] | None = None,
        workers: int = 1,
    ):

  # Convert input/output paths to Path objects
  source, output = Path(responses_path), Path(output_path)
  if source.resolve() == output.resolve():
    raise ValueError("Vector output must differ from response input.")
  if workers < 1:
    raise ValueError("workers must be at least 1.")
  if workers > 1 and diagnostics:
    raise ValueError("Parallel vector generation does not support --diagnostics.")
  if workers > 1 and score_pair is not None:
    raise ValueError("Parallel vector generation does not support a custom score_pair.")

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
                      root / "SLM" / "embedding.py",
                      *sorted((root / "TextProcessing").glob("*.py"))]

  # Fingerprint inputs, scorer code, and settings so resume cannot mix experiments.
  metadata = {"schema_version": 1, "response_sha256": file_digest(source),
              "response_file": str(source.resolve()), "essay_files": essays, "response_count": count,
              "algorithm_sha256": {str(p.relative_to(root)): file_digest(p) for p in algorithm_files},
              "parser_dir": str(Path(parser_dir).resolve()), "diagnostics": diagnostics,
              "embeddings_model": configuration.embeddings_model,
              "bertscore_device": configuration.bertscore_device,
              "bertscore_pair_batch_size": configuration.bertscore_pair_batch_size,
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

  # Build all unfinished pair lists before dispatching any worker. Their order is the
  # canonical CSV order, which the parent writer preserves in parallel mode.
  work_by_essay: list[tuple[str, list[PairWork]]] = []
  for essay in essays:
    unfinished_pairs = []
    for p, q in [("A", "A"), ("B", "B"), ("A", "B")]:
      ids = range(1, count + 1)
      pairs = product(ids, ids) if p != q else combinations_with_replacement(ids, 2)
      for i, j in pairs:
        key = (essay, p, i, q, j)
        if key not in completed:
          unfinished_pairs.append((key, lookup[(essay, p, i)], lookup[(essay, q, j)]))
    if unfinished_pairs:
      work_by_essay.append((essay, unfinished_pairs))

  owned_scorer = None
  try:
    with output.open("x" if fresh else "a", encoding="utf-8", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=OLMSVectorTable.COLUMNS)
      if fresh:
        writer.writeheader()
        handle.flush()

      if workers > 1:
        _score_parallel(
          work_by_essay, parser_dir, workers, handle, writer, count, completed, total)
      else:
        # The one-worker path retains the existing scorer lifecycle and direct test seam.
        for essay, unfinished_pairs in work_by_essay:
          if score_pair is None:
            from scoring import OLMSPairScorer
            owned_scorer = OLMSPairScorer(parser_dir, diagnostics_dir)
            score_pair = owned_scorer
          if owned_scorer is not None:
            owned_scorer.prepare_embeddings([
              response for _, first, second in unfinished_pairs
              for response in (first, second)
            ])
            scores = owned_scorer.score_pairs([
              (first, second) for _, first, second in unfinished_pairs
            ])
            for (key, _, _), score in zip(unfinished_pairs, scores, strict=True):
              _append_vector_row(handle, writer, key, score, count, completed, total)
          else:
            for key, first, second in unfinished_pairs:
              _append_vector_row(
                handle, writer, key, score_pair(first, second), count, completed, total)

    # Revalidate the finished table before marking vector metadata complete.
    OLMSVectorTable(output)
    metadata["complete"] = True
    write_metadata(metadata_path(output), metadata)
    return output
  finally:
    if owned_scorer is not None:
      owned_scorer.close()
