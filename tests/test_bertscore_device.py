import unittest
from unittest.mock import patch

import configuration
from TextProcessing import shared_bert_scorer


class BertScoreDeviceTests(unittest.TestCase):
  def setUp(self):
    shared_bert_scorer.get_bert_scorer.cache_clear()
    self.addCleanup(shared_bert_scorer.get_bert_scorer.cache_clear)

  @patch("TextProcessing.shared_bert_scorer.BERTScorer")
  @patch.object(configuration, "bertscore_device", "auto")
  @patch("TextProcessing.shared_bert_scorer.torch.backends.mps.is_available", return_value=True)
  @patch("TextProcessing.shared_bert_scorer.torch.backends.mps.is_built", return_value=True)
  def test_auto_uses_mps_when_available(self, built, available, scorer):
    shared_bert_scorer.get_bert_scorer()
    scorer.assert_called_once_with(lang="en", rescale_with_baseline=True, device="mps")

  @patch("TextProcessing.shared_bert_scorer.BERTScorer")
  @patch.object(configuration, "bertscore_device", "auto")
  @patch("TextProcessing.shared_bert_scorer.torch.backends.mps.is_available", return_value=False)
  @patch("TextProcessing.shared_bert_scorer.torch.backends.mps.is_built", return_value=True)
  def test_auto_falls_back_to_cpu_when_mps_is_unavailable(self, built, available, scorer):
    shared_bert_scorer.get_bert_scorer()
    scorer.assert_called_once_with(lang="en", rescale_with_baseline=True, device="cpu")

  @patch.object(configuration, "bertscore_device", "mps")
  @patch("TextProcessing.shared_bert_scorer.torch.backends.mps.is_available", return_value=False)
  @patch("TextProcessing.shared_bert_scorer.torch.backends.mps.is_built", return_value=True)
  def test_explicit_mps_rejects_an_unavailable_backend(self, built, available):
    with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
      shared_bert_scorer.resolve_bertscore_device()

  @patch.object(configuration, "bertscore_device", "other")
  def test_rejects_unknown_device_configuration(self):
    with self.assertRaisesRegex(ValueError, "must be"):
      shared_bert_scorer.resolve_bertscore_device()


if __name__ == "__main__":
  unittest.main()
