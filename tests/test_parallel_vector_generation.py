import csv
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysis.olms_vector_table import OLMSVectorTable
from vector_generation import _score_parallel, generate_vectors


class ClosableQueue(queue.Queue):
  def close(self):
    pass


class ThreadProcess:
  def __init__(self, target, args):
    self.thread = threading.Thread(target=target, args=args)

  def start(self):
    self.thread.start()

  @property
  def exitcode(self):
    return 0 if not self.thread.is_alive() else None

  def is_alive(self):
    return self.thread.is_alive()

  def terminate(self):
    raise AssertionError("A successful parallel worker must not be terminated.")

  def join(self):
    self.thread.join()


class ThreadContext:
  def Queue(self):
    return ClosableQueue()

  def Process(self, target, args):
    return ThreadProcess(target, args)


class FakeScorer:
  def __init__(self, parser_dir):
    self.parser_dir = parser_dir

  def prepare_embeddings(self, responses):
    pass

  def __call__(self, first, second):
    # Ensure the second essay reports first, exercising the parent's reorder buffer.
    if first["essay_file"] == "first.md":
      time.sleep(0.02)
    return {"org": 0.8, "lex": 0.7, "meaning": 0.9, "struct": 0.6}

  def close(self):
    pass


class ParallelVectorGenerationTests(unittest.TestCase):
  @staticmethod
  def pair(essay):
    first = {"essay_file": essay, "prompt": "A", "response_index": 1, "response": "First"}
    second = {"essay_file": essay, "prompt": "B", "response_index": 1, "response": "Second"}
    return ((essay, "A", 1, "B", 1), first, second)

  def test_parallel_writer_preserves_essay_order(self):
    fake_scoring = types.ModuleType("scoring")
    fake_scoring.OLMSPairScorer = FakeScorer
    work = [("first.md", [self.pair("first.md")]), ("second.md", [self.pair("second.md")])]

    with tempfile.TemporaryDirectory() as directory, \
         patch.dict(sys.modules, {"scoring": fake_scoring}), \
         patch("vector_generation.multiprocessing.get_context", return_value=ThreadContext()):
      output = Path(directory) / "vectors.csv"
      with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OLMSVectorTable.COLUMNS)
        writer.writeheader()
        completed = set()
        _score_parallel(work, ".models/udpipe", 2, handle, writer, 2, completed, 2)

      with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
      self.assertEqual([row["essay_file"] for row in rows], ["first.md", "second.md"])
      self.assertEqual(len(completed), 2)

  def test_two_worker_csv_matches_serial_output(self):
    fake_scoring = types.ModuleType("scoring")
    fake_scoring.OLMSPairScorer = FakeScorer
    with tempfile.TemporaryDirectory() as directory, \
         patch.dict(sys.modules, {"scoring": fake_scoring}), \
         patch("vector_generation.multiprocessing.get_context", return_value=ThreadContext()):
      root = Path(directory)
      responses = root / "responses.csv"
      parallel = root / "parallel.csv"
      serial = root / "serial.csv"
      with responses.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
          handle, fieldnames=["essay_file", "prompt", "response_index", "response"])
        writer.writeheader()
        writer.writerows(
          {"essay_file": essay, "prompt": prompt, "response_index": index,
           "response": f"{essay} {prompt}{index}"}
          for essay in ["first.md", "second.md"]
          for prompt in ["A", "B"]
          for index in [1, 2]
        )

      generate_vectors(responses, parallel, parser_dir=".models/udpipe", workers=2)
      generate_vectors(
        responses, serial, parser_dir=".models/udpipe",
        score_pair=lambda first, second: {"org": 0.8, "lex": 0.7, "meaning": 0.9, "struct": 0.6},
      )

      pd.testing.assert_frame_equal(pd.read_csv(parallel), pd.read_csv(serial))
      OLMSVectorTable(parallel)


if __name__ == "__main__":
  unittest.main()
