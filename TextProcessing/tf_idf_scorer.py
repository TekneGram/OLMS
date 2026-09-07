import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFScore:
  def __init__(
      self,
      text_a: list[str],
      text_b: list[str]
  ) -> None:
    self.text_a = text_a
    self.text_b = text_b

  def create_tfidf_score_table(self) -> pd.DataFrame:
    if not self.text_a or not self.text_b:
      return pd.DataFrame(index=self.text_a, columns=self.text_b)

    vectorizer = TfidfVectorizer()

    try:
      vectorizer.fit(self.text_a + self.text_b)
    except ValueError:
      return pd.DataFrame(
        [
          [
            0.0
            for _ in self.text_b
          ]
          for _ in self.text_a
        ],
        index=self.text_a,
        columns=self.text_b
      )

    vectors_a = vectorizer.transform(self.text_a)
    vectors_b = vectorizer.transform(self.text_b)
    scores = cosine_similarity(vectors_a, vectors_b)

    return pd.DataFrame(
      scores,
      index=self.text_a,
      columns=self.text_b
    )
