import unittest

from OLMSClasses.L import L, Phraseology
from TextProcessing.phrase_ngrams import PhraseNGramExtractor


class PhraseNGramExtractorTests(unittest.TestCase):
  def setUp(self):
    self.extractor = PhraseNGramExtractor()

  def test_extracts_lowercase_ngrams_without_crossing_sentences(self):
    text = "Students need evidence. This supports the claim."

    self.assertEqual(
      self.extractor.counts(text, 2),
      {
        "students need": 1,
        "need evidence": 1,
        "this supports": 1,
        "supports the": 1,
        "the claim": 1,
      },
    )
    self.assertNotIn("evidence this", self.extractor.counts(text, 2))

  def test_rejects_unsupported_ngram_sizes(self):
    with self.assertRaises(ValueError):
      self.extractor.counts("some text", 4)


class PhraseologyTests(unittest.TestCase):
  def test_identical_text_has_maximum_similarity(self):
    phraseology = Phraseology(
      "Students need evidence to support claims.",
      "Students need evidence to support claims.",
    )

    self.assertAlmostEqual(phraseology.jsd_2gram(), 0.0)
    self.assertAlmostEqual(phraseology.jsd_3gram(), 0.0)
    self.assertAlmostEqual(phraseology.similarity(), 1.0)

  def test_different_phrase_distributions_have_lower_similarity(self):
    phraseology = Phraseology(
      "Students need evidence to support claims.",
      "The weather changed quickly during winter.",
    )

    self.assertGreater(phraseology.jsd_2gram(), 0.0)
    self.assertGreater(phraseology.jsd_3gram(), 0.0)
    self.assertLess(phraseology.similarity(), 1.0)

  def test_short_text_returns_no_phraseology_evidence(self):
    phraseology = Phraseology("Text.", "Text.")

    self.assertEqual(phraseology.similarity(), 0.0)

  def test_lexical_similarity_includes_phraseology_at_twenty_percent(self):
    lexical = L("Students need evidence to support claims.",
                 "Students need evidence to support claims.")

    self.assertAlmostEqual(lexical.lexical_similarity(), 1.0)


if __name__ == "__main__":
  unittest.main()
