from collections import Counter
import re


class PhraseNGramExtractor:
  """Extract lowercase, sentence-bounded word n-gram counts from raw text."""

  def counts(self, text: str, n: int) -> Counter[str]:
    if n not in (2, 3):
      raise ValueError("Only 2-grams and 3-grams are supported.")

    # Raw text keeps this scorer independent of the dependency parser used by O/S.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.lower().strip())
    ngrams: Counter[str] = Counter()

    for sentence in sentences:
      tokens = re.findall(r"\b\w+\b", sentence)
      ngrams.update(
        " ".join(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
      )

    return ngrams
