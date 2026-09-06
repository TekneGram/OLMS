from bert_score import BERTScorer
import pandas as pd

class BertScore:
  def __init__(
      self,
      text_a: list[str],
      text_b: list[str]
  ) -> None:
    self.text_a = text_a
    self.text_b = text_b
    self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)

  def create_bert_score_table(self) -> pd.DataFrame:
    pairs = [(x, y) for x in self.text_a for y in self.text_b]
    candidates = [x for x, y in pairs]
    references = [y for x, y, in pairs]

    precision, recall, f1 = self.scorer.score(candidates, references)

    precision = precision.reshape(len(self.text_a), len(self.text_b))
    recall = recall.reshape(len(self.text_a), len(self.text_b))
    f1 = f1.reshape(len(self.text_a), len(self.text_b))

    table = pd.DataFrame(
      [
        [
          [
            precision[i, j].item(),
            recall[i, j].item(),
            f1[i, j].item(),
          ]
          for j in range(len(self.text_b))
        ]
        for i in range(len(self.text_a))
      ],
      index=self.text_a,
      columns=self.text_b
    )
    return table