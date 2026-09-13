"""Essay-local contextual-token cache for BERTScore.

The upstream ``BERTScorer.score`` helper re-encodes every text supplied to a
call.  OLMS compares the same responses and parsed clauses many times, so this
cache retains the unpadded contextual embeddings and IDF weights for each
distinct text.  Greedy matching deliberately still uses BERTScore's own
implementation.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from bert_score.utils import get_bert_embedding, greedy_cos_idf

from TextProcessing.shared_bert_scorer import clear_bertscore_memory


def _is_out_of_memory(error: RuntimeError) -> bool:
  message = str(error).lower()
  return "out of memory" in message or "not enough memory" in message


class BertScoreContextCache:
  """Cache raw BERTScore token statistics for one essay.

  Stored tensors live on CPU, just as BERTScore's own ``bert_cos_score_idf``
  implementation stores its per-text statistics.  This bounds accelerator
  memory while allowing the matching step to move only its current pair batch
  back to the configured device.
  """

  ENCODING_BATCH_SIZE = 64

  def __init__(self, scorer) -> None:
    self.scorer = scorer
    self._stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

  def clear(self) -> None:
    self._stats.clear()

  def prepare(self, texts: Sequence[str]) -> None:
    """Encode each missing distinct text once, in BERTScore-compatible chunks."""
    missing = sorted(
      {text for text in texts if text not in self._stats},
      key=lambda text: (-len(text.split(" ")), text),
    )
    for start in range(0, len(missing), self.ENCODING_BATCH_SIZE):
      self._prepare_chunk_with_backoff(missing[start:start + self.ENCODING_BATCH_SIZE])

  def _prepare_chunk_with_backoff(self, texts: Sequence[str]) -> None:
    try:
      self._prepare_chunk(texts)
    except RuntimeError as error:
      if not _is_out_of_memory(error) or len(texts) == 1:
        raise
      clear_bertscore_memory()
      midpoint = len(texts) // 2
      self._prepare_chunk_with_backoff(texts[:midpoint])
      self._prepare_chunk_with_backoff(texts[midpoint:])

  def _prepare_chunk(self, texts: Sequence[str]) -> None:
    if not texts:
      return
    embeddings, masks, padded_idf = get_bert_embedding(
      list(texts),
      self.scorer._model,
      self.scorer._tokenizer,
      self._idf_dict(),
      device=self.scorer.device,
      all_layers=self.scorer.all_layers,
    )
    embeddings = embeddings.cpu()
    masks = masks.cpu()
    padded_idf = padded_idf.cpu()
    for index, text in enumerate(texts):
      length = masks[index].sum().item()
      # greedy_cos_idf normalizes its padded inputs in place.  Keep independent,
      # immutable source tensors so the same response can be matched repeatedly.
      self._stats[text] = (
        embeddings[index, :length].detach().clone(),
        padded_idf[index, :length].detach().clone(),
      )

  def score(
      self,
      candidates: Sequence[str],
      references: Sequence[str],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BERTScore precision, recall, and F1 for aligned text pairs."""
    if len(candidates) != len(references):
      raise ValueError("BERTScore requires equally sized candidate and reference lists.")
    if not candidates:
      empty = torch.empty(0)
      return empty, empty, empty

    self.prepare([*candidates, *references])
    reference_stats = self._pad_batch_stats(references)
    candidate_stats = self._pad_batch_stats(candidates)
    with torch.no_grad():
      precision, recall, f1 = greedy_cos_idf(
        *reference_stats, *candidate_stats, all_layers=self.scorer.all_layers)
      scores = torch.stack((precision, recall, f1), dim=-1).cpu()

    if self.scorer.rescale_with_baseline:
      scores = (scores - self.scorer.baseline_vals) / (1 - self.scorer.baseline_vals)
    return scores[..., 0], scores[..., 1], scores[..., 2]

  def _idf_dict(self):
    if self.scorer.idf:
      if not self.scorer._idf_dict:
        raise ValueError("BERTScore IDF weights are not computed.")
      return self.scorer._idf_dict
    idf_dict = defaultdict(lambda: 1.0)
    idf_dict[self.scorer._tokenizer.sep_token_id] = 0
    idf_dict[self.scorer._tokenizer.cls_token_id] = 0
    return idf_dict

  def _pad_batch_stats(
      self,
      texts: Sequence[str],
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings, idfs = zip(*(self._stats[text] for text in texts))
    device = next(self.scorer._model.parameters()).device
    # The clones protect cached source values if a CPU execution path aliases a
    # tensor passed to an in-place BERTScore operation.
    embeddings = [embedding.clone().to(device) for embedding in embeddings]
    idfs = [idf.clone().to(device) for idf in idfs]
    lengths = [embedding.size(0) for embedding in embeddings]
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=2.0)
    padded_idfs = pad_sequence(idfs, batch_first=True)
    maximum = max(lengths)
    positions = torch.arange(maximum, dtype=torch.long).expand(len(lengths), maximum)
    mask = positions < torch.tensor(lengths, dtype=torch.long).unsqueeze(1)
    return padded_embeddings, mask.to(device), padded_idfs
