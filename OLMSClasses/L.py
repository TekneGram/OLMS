# This script calculates a lexical similarity score using TF-IDF cosine similarity.
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from TextProcessing.phrase_ngrams import PhraseNGramExtractor


class Phraseology:
  """Compare sentence-bounded 2-gram and 3-gram frequency distributions."""

  def __init__(
      self,
      text_a: str,
      text_b: str,
      smoothing: float = 1e-9,
  ) -> None:
    self.text_a = text_a
    self.text_b = text_b
    self.smoothing = smoothing
    self.extractor = PhraseNGramExtractor()

  def _jsd(self, counts_a: Counter[str], counts_b: Counter[str]) -> float:
    if not counts_a or not counts_b:
      # No shared evidence exists when either response has no usable n-grams.
      return 1.0

    vocabulary = set(counts_a) | set(counts_b)
    total_a = sum(counts_a[phrase] + self.smoothing for phrase in vocabulary)
    total_b = sum(counts_b[phrase] + self.smoothing for phrase in vocabulary)

    distribution_a = {
      phrase: (counts_a[phrase] + self.smoothing) / total_a
      for phrase in vocabulary
    }
    distribution_b = {
      phrase: (counts_b[phrase] + self.smoothing) / total_b
      for phrase in vocabulary
    }
    midpoint = {
      phrase: (distribution_a[phrase] + distribution_b[phrase]) / 2.0
      for phrase in vocabulary
    }

    def kl_divergence(distribution: dict[str, float]) -> float:
      return sum(
        probability * math.log2(probability / midpoint[phrase])
        for phrase, probability in distribution.items()
      )

    # Smoothing prevents log(0) and gives absent phrases a small probability.
    divergence = (
      kl_divergence(distribution_a) + kl_divergence(distribution_b)
    ) / 2.0
    return min(1.0, max(0.0, divergence))

  def jsd_2gram(self) -> float:
    return self._jsd(
      self.extractor.counts(self.text_a, 2),
      self.extractor.counts(self.text_b, 2),
    )

  def jsd_3gram(self) -> float:
    return self._jsd(
      self.extractor.counts(self.text_a, 3),
      self.extractor.counts(self.text_b, 3),
    )

  def similarity_2gram(self) -> float:
    return 1.0 - self.jsd_2gram()

  def similarity_3gram(self) -> float:
    return 1.0 - self.jsd_3gram()

  def similarity(self) -> float:
    # Equal weighting keeps the shorter 2-gram and more specific 3-gram views balanced.
    return (self.similarity_2gram() + self.similarity_3gram()) / 2.0

class L:
  """
  Lexical similarity between two full small language model responses.
  Implements:
    S_TF = cosine_similarity(TF_IDF(A), TF_IDF(B))

  where A and B are original full small language model responses.
  """
  def __init__(
      self,
      text_a: str,
      text_b: str,
  ) -> None:
    self.text_a = text_a.strip()
    self.text_b = text_b.strip()

  def tf_idf_score(self) -> float:
    """
    Return TF-IDF cosine similarity between the two texts
    Returns a value in [0, 1]
    """

    if not self.text_a or not self.text_b:
      return 0.0

    vectorizer = TfidfVectorizer()

    try:
      vectors = vectorizer.fit_transform([
        self.text_a,
        self.text_b
      ])
    except ValueError:
      return 0.0

    score = cosine_similarity(
      vectors[0],
      vectors[1]
    )[0, 0]
    
    return float(score)

  def _tokenize_words(self, text:str) -> list[str]:
    """
    Tokenize text into lowercase word tokens
    """
    return re.findall(r"\b\w+\b", text.lower())

  # ---------
  # Measure of Textual Lexical Diversity Area
  # ---------

  def _mtld(
      self,
      text: str,
      threshold: float = 0.72
  ) -> float:
    """
    Calculate Measure of Textual Lexical Diversity

    Higher MTLD means greater lexical richness
    MTLD is computed in both forward and backward directions, then averaged
    """

    tokens = self._tokenize_words(text)

    if not tokens:
      return 0.0

    return(
      self._mtld_direction(tokens, threshold) + self._mtld_direction(list(reversed(tokens)), threshold)
    ) / 2.0

  def _mtld_direction(
      self,
      tokens: list[str],
      threshold: float
  ) -> float:
    """
    Calculate MTLD in one direction
    """

    factors = 0.0
    types = set()
    token_count = 0

    for token in tokens:
      token_count += 1
      types.add(token)

      ttr = len(types) / token_count

      if ttr <= threshold:
        factors += 1.0
        types = set()
        token_count = 0

    # Partial factor for remaining tokens
    if token_count > 0:
      ttr = len(types) / token_count

      if ttr == 1.0:
        partial_factor = 0.0
      else:
        partial_factor = (1.0 - ttr) / (1.0 - threshold)

      factors += partial_factor

    if factors == 0.0:
      return float(len(tokens))

    return len(tokens) / factors

  def mtld_similarity(
      self,
      threshold: float = 0.72,
  ) -> float:
    """
    Compare two texts by MTLD similarity
    Returns a value in [0,1]
    1.0 = same MTLD
    0.0 = maximally different lexical diversity
    """
    mtld_a = self._mtld(self.text_a, threshold)
    mtld_b = self._mtld(self.text_b, threshold)

    max_mtld = max(mtld_a, mtld_b)

    if max_mtld == 0.0:
      return 0.0

    return 1.0 - abs(mtld_a - mtld_b) / max_mtld

  # ------------
  # Moving Average Type-Token Ratio
  # ------------

  def _mattr(
      self,
      text: str,
      window_size: int = 25,
  ) -> float:
    """
    Calculate Moving Average Type-Token Ratio
    Returna. value in [0, 1]
    Higher means greater local lexical diversity
    """

    tokens = self._tokenize_words(text)

    if not tokens:
      return 0.0

    if len(tokens) <= window_size:
      return len(set(tokens)) / len(tokens)

    window_scores = []

    for start in range(len(tokens) - window_size + 1):
      window = tokens[start:start + window_size]
      window_scores.append(
        len(set(window)) / window_size
      )

    return float(sum(window_scores) / len(window_scores))

  def mattr_similarity(
      self,
      window_size: int = 25,
  ) -> float:
    """
    Compare two texts by MATTR similarity

    Returns a value in [0, 1]
    1.0 = same MATTR
    0.0 = maximally different local lexical diversity
    """

    mattr_a = self._mattr(self.text_a, window_size)
    mattr_b = self._mattr(self.text_b, window_size)

    return 1.0 - abs(mattr_a - mattr_b)

  def lexical_similarity(
      self,
      text_size: int = 99,
      window_size: int = 25,
      threshold: float = 0.72,
  ) -> float:
    if text_size >= 100:
      current_lexical_similarity = (self.mtld_similarity(threshold) + self.tf_idf_score())/2.0
    else:
      current_lexical_similarity = (self.mattr_similarity(window_size) + self.tf_idf_score())/2.0

    phraseology_similarity = Phraseology(
      self.text_a,
      self.text_b,
    ).similarity()

    # Phraseology is an additional signal, so vocabulary remains the main L component.
    return current_lexical_similarity * 0.8 + phraseology_similarity * 0.2
