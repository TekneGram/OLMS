from pathlib import Path

from TextProcessing.deprel import DependencyParse, DependencyToken


class SentenceSplitter:
  def __init__(self, parsed: DependencyParse) -> None:
    self.parsed = parsed

  def split_all(self) -> list[str]:
    return [
      self.token_text(sentence_tokens)
      for sentence_tokens in self.split_all_tokens()
    ]

  def split_all_tokens(self) -> list[list[DependencyToken]]:
    return [
      self.parsed.tokens_by_sentence[sentence_id]
      for sentence_id in sorted(self.parsed.tokens_by_sentence)
    ]

  @staticmethod
  def append_clause_sets_to_markdown(
      output_path: str | Path,
      filename: str,
      clause_sets: dict[str, list[str]],
  ) -> None:
    output_path = Path(output_path)

    with output_path.open("a", encoding="utf-8") as file:
      file.write(f"## Filename: {filename}\n\n")

      for label, clauses in clause_sets.items():
        file.write(f"### {label}\n\n")

        for index, clause in enumerate(clauses, start=1):
          file.write(f"{index}. {clause}\n")

        file.write("\n")

  def split_sentence(
      self,
      sentence_tokens: list[DependencyToken],
  ) -> list[str]:
    return [self.token_text(sentence_tokens)] if sentence_tokens else []

  def split_sentence_tokens(
      self,
      sentence_tokens: list[DependencyToken],
  ) -> list[list[DependencyToken]]:
    return [sentence_tokens] if sentence_tokens else []

  def token_text(self, tokens: list[DependencyToken]) -> str:
    tokens = sorted(tokens, key=lambda token: token.token_id)
    words = [token.text for token in tokens if token.upos != "PUNCT"]
    text = " ".join(words)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text
