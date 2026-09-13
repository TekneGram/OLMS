import tempfile
import unittest
from pathlib import Path

from calibration.calibrate import extract_sections, run_calibration


class FakeScorer:
  def __init__(self, parser_dir):
    self.parser_dir = parser_dir
    self.pairs = []
    self.closed = False

  def score_pair(self, first, second):
    self.pairs.append((first, second))
    return {"org": 0.1, "lex": 0.2, "meaning": 0.3, "struct": 0.4}

  def close(self):
    self.closed = True


class CalibrationTests(unittest.TestCase):
  def test_extract_sections_excludes_headings_and_preserves_paragraphs(self):
    sections = extract_sections(
      "## I\nFirst paragraph.\n\nSecond paragraph.\n\n"
      "## II\nSecond section.\n\n## III\nThird section.\n\n## IV\nFourth section."
    )

    self.assertEqual(sections["I"], "First paragraph.\n\nSecond paragraph.")
    self.assertEqual(sections["II"], "Second section.")
    self.assertEqual(sections["III"], "Third section.")
    self.assertEqual(sections["IV"], "Fourth section.")
    self.assertNotIn("## I", sections["I"])

  def test_extract_sections_rejects_missing_and_duplicate_headings(self):
    with self.assertRaisesRegex(ValueError, "Missing"):
      extract_sections("## I\nText\n\n## II\nText")

    with self.assertRaisesRegex(ValueError, "Duplicate"):
      extract_sections("## I\nOne\n\n## I\nTwo\n\n## II\nTwo\n\n## III\nThree\n\n## IV\nFour")

  def test_run_calibration_writes_six_comparisons_per_file(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      texts = root / "texts"
      texts.mkdir()
      (texts / "A.md").write_text(
        "## I\nOne two three.\n\n## II\nTwo three four.\n\n## III\nThree four five.\n\n## IV\nFour five six.",
        encoding="utf-8",
      )
      (texts / "B.md").write_text(
        "## I\nOne two three.\n\n## II\nTwo three four.\n\n## III\nThree four five.\n\n## IV\nFour five six.",
        encoding="utf-8",
      )

      scorers = []

      def factory(parser_dir):
        scorer = FakeScorer(parser_dir)
        scorers.append(scorer)
        return scorer

      output = run_calibration(texts, root / "calibration_results.md", scorer_factory=factory)
      report = output.read_text(encoding="utf-8")

      self.assertEqual(len(scorers), 1)
      self.assertTrue(scorers[0].closed)
      self.assertEqual(len(scorers[0].pairs), 12)
      self.assertIn("## A.md", report)
      self.assertIn("## B.md", report)
      self.assertEqual(report.count("| I-II |"), 2)
      self.assertEqual(report.count("| I-III |"), 2)
      self.assertEqual(report.count("| II-III |"), 2)
      self.assertEqual(report.count("| I-IV |"), 2)
      self.assertEqual(report.count("| II-IV |"), 2)
      self.assertEqual(report.count("| III-IV |"), 2)
      self.assertIn("0.100000 | 0.200000 | 0.300000 | 0.400000", report)

      rows = [
        line for line in report.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
        and not line.startswith("| Comparison")
      ]
      self.assertEqual(
        [row.split("|")[1].strip() for row in rows],
        ["I-II", "I-III", "II-III", "I-IV", "II-IV", "III-IV"] * 2,
      )


if __name__ == "__main__":
  unittest.main()
