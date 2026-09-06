# This script calculates a lexical similarity score using TF-IDF cosine similarity.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

  def lexical_score(self) -> float:
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