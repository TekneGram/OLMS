# This script calculates a lexical similarity score using TF-IDF cosine similarity.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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