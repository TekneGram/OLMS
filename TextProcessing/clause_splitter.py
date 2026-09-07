from TextProcessing.deprel import DependencyToken, DependencyParse
from pathlib import Path

class ClauseSplitter:
  SPLIT_DEPRELS = {"conj", "advcl", "acl:relcl"}
  CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
  PREDICATE_UPOS = {"VERB", "ADJ", "NOUN"}
  NOMINAL_UPOS = {"NOUN", "PROPN", "PRON"}
  MIN_STRICT_NONPUNCT_TOKENS = 5
  MIN_STRICT_CONTENT_TOKENS = 3
  WEAK_FRAGMENT_STARTS = {
    "and",
    "or",
    "but",
    "that",
    "which",
    "because",
    "to",
    "by",
    "with",
  }
  DISCOURSE_ONLY_TEXTS = {
    "for example",
    "however",
  }

  def __init__(self, parsed: DependencyParse) -> None:
    self.parsed = parsed

  def split_all(self) -> list[str]:
    return [
      self.token_text(clause_tokens)
      for clause_tokens in self.split_all_tokens()
    ]

  def split_all_tokens(self) -> list[list[DependencyToken]]:
    clauses = []

    for sentence_tokens in self.parsed.tokens_by_sentence.values():
      clauses.extend(self.split_sentence_tokens(sentence_tokens))

    return clauses

  def split_all_strict(self) -> list[str]:
    return [
      self.token_text(clause_tokens)
      for clause_tokens in self.split_all_tokens_strict()
    ]

  def split_all_tokens_strict(self) -> list[list[DependencyToken]]:
    clauses = []

    for sentence_tokens in self.parsed.tokens_by_sentence.values():
      clauses.extend(self.split_sentence_tokens_strict(sentence_tokens))

    return self.merge_strict_fragments(clauses)

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
    return [
      self.token_text(clause_tokens)
      for clause_tokens in self.split_sentence_tokens(sentence_tokens)
    ]

  def split_sentence_strict(
      self,
      sentence_tokens: list[DependencyToken],
  ) -> list[str]:
    return [
      self.token_text(clause_tokens)
      for clause_tokens in self.split_sentence_tokens_strict(sentence_tokens)
    ]

  def clause_start_id(self, tokens: list[DependencyToken]) -> int:
      return min(token.token_id for token in tokens)

  def split_sentence_tokens(
      self,
      sentence_tokens: list[DependencyToken],
  ) -> list[list[DependencyToken]]:
    if not sentence_tokens:
      return []

    sentence_id = sentence_tokens[0].sentence_id
    split_roots = []

    for token in sentence_tokens:
      if token.dependency_relation not in self.SPLIT_DEPRELS:
        continue

      subtree = self.collect_subtree(sentence_id, token.token_id)

      if self.is_clause_like(token, subtree):
        split_roots.append(token)

    split_root_ids = {root.token_id for root in split_roots}

    main_clause = [
      token for token in sentence_tokens
      if token.token_id not in split_root_ids
      and not self.is_descendent_of_any_split(token, split_roots)
    ]

    split_clauses = []

    if main_clause:
      split_clauses.append(main_clause)

    for root in split_roots:
      split_clauses.append(
        self.collect_subtree(sentence_id, root.token_id)
      )

    split_clauses = sorted(
      split_clauses,
      key=self.clause_start_id
    )

    return split_clauses

  def split_sentence_tokens_strict(
      self,
      sentence_tokens: list[DependencyToken],
  ) -> list[list[DependencyToken]]:
    if not sentence_tokens:
      return []

    sentence_id = sentence_tokens[0].sentence_id
    split_roots = []

    for token in sentence_tokens:
      if token.dependency_relation not in self.SPLIT_DEPRELS:
        continue

      subtree = self.collect_subtree(sentence_id, token.token_id)

      if self.is_viable_strict_split(token, subtree):
        split_roots.append(token)

    split_roots = self.filter_nested_strict_roots(split_roots)
    split_root_ids = {root.token_id for root in split_roots}

    main_clause = [
      token for token in sentence_tokens
      if token.token_id not in split_root_ids
      and not self.is_descendent_of_any_split(token, split_roots)
    ]

    split_clauses = []

    if main_clause and not self.is_discourse_only_fragment(main_clause):
      split_clauses.append(main_clause)

    for root in split_roots:
      clause = self.collect_subtree(sentence_id, root.token_id)
      if not self.is_discourse_only_fragment(clause):
        split_clauses.append(clause)

    split_clauses = sorted(
      split_clauses,
      key=self.clause_start_id
    )

    return split_clauses

  def collect_subtree(
      self,
      sentence_id: int,
      root_token_id: int
  ) -> list[DependencyToken]:
    root = self.parsed.token_by_sentence_and_id.get(
      (sentence_id, root_token_id)
    )

    result = [root] if root is not None else []

    def visit(token_id: int) -> None:
      children = self.parsed.children_by_sentence_and_head.get(
        (sentence_id, token_id),
        []
      )

      for child in children:
        result.append(child)
        visit(child.token_id)

    visit(root_token_id)
    return result

  def token_text(self, tokens: list[DependencyToken]) -> str:
    tokens = sorted(tokens, key=lambda token: token.global_token_id)
    words = [token.text for token in tokens if token.upos != "PUNCT"]
    text = " ".join(words)
    text = text.replace(" ,", ",").replace( " .", ".")
    return text

  def nonpunct_tokens(
      self,
      tokens: list[DependencyToken]
  ) -> list[DependencyToken]:
    return [token for token in tokens if token.upos != "PUNCT"]

  def content_tokens(
      self,
      tokens: list[DependencyToken]
  ) -> list[DependencyToken]:
    return [token for token in tokens if token.upos in self.CONTENT_UPOS]

  def is_clause_like(
      self,
      root: DependencyToken,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    nonpunct = self.nonpunct_tokens(subtree_tokens)
    content = self.content_tokens(subtree_tokens)

    has_predicate_root = root.upos == "VERB"

    has_subject = any(
      token.dependency_relation in {"nsubj", "nsubj:pass"}
      for token in subtree_tokens
    )

    subject_ok = has_subject or root.dependency_relation == "conj"

    return (
      has_predicate_root
      and subject_ok
      and len(nonpunct) >= 4
      and len(content) >= 2
    )

  def is_viable_strict_split(
      self,
      root: DependencyToken,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    if not self.is_strict_clause_like(root, subtree_tokens):
      return False

    if self.is_discourse_only_fragment(subtree_tokens):
      return False

    if self.is_weak_fragment(root, subtree_tokens):
      return False

    if root.dependency_relation == "advcl":
      return not self.is_infinitival_fragment(root, subtree_tokens)

    if root.dependency_relation == "acl:relcl":
      return self.is_viable_relative_clause(subtree_tokens)

    return True

  def is_strict_clause_like(
      self,
      root: DependencyToken,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    nonpunct = self.nonpunct_tokens(subtree_tokens)
    content = self.content_tokens(subtree_tokens)

    has_subject = any(
      token.dependency_relation in {"nsubj", "nsubj:pass"}
      for token in subtree_tokens
    )

    subject_ok = has_subject or root.dependency_relation == "conj"

    return (
      self.has_predicate_root(root, subtree_tokens)
      and subject_ok
      and len(nonpunct) >= self.MIN_STRICT_NONPUNCT_TOKENS
      and len(content) >= self.MIN_STRICT_CONTENT_TOKENS
      and self.has_nominal_anchor(subtree_tokens)
    )

  def has_predicate_root(
      self,
      root: DependencyToken,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    if root.upos == "VERB":
      return True

    if root.upos not in {"ADJ", "NOUN"}:
      return False

    return any(
      token.dependency_relation in {"cop", "aux:pass"}
      for token in subtree_tokens
    )

  def has_nominal_anchor(
      self,
      tokens: list[DependencyToken],
  ) -> bool:
    return any(
      token.upos in self.NOMINAL_UPOS
      for token in tokens
    )

  def first_nonpunct_token(
      self,
      tokens: list[DependencyToken],
  ) -> DependencyToken | None:
    nonpunct = self.nonpunct_tokens(tokens)

    if not nonpunct:
      return None

    return sorted(nonpunct, key=lambda token: token.token_id)[0]

  def is_discourse_only_fragment(
      self,
      tokens: list[DependencyToken],
  ) -> bool:
    text = self.token_text(tokens).lower()
    return text in self.DISCOURSE_ONLY_TEXTS

  def is_weak_fragment(
      self,
      root: DependencyToken,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    first = self.first_nonpunct_token(subtree_tokens)

    if first is None:
      return True

    if first.text.lower() not in self.WEAK_FRAGMENT_STARTS:
      return False

    has_explicit_subject = any(
      token.dependency_relation in {"nsubj", "nsubj:pass"}
      for token in subtree_tokens
    )

    if first.text.lower() in {"and", "or", "but"}:
      return not has_explicit_subject

    return True

  def should_merge_with_previous(
      self,
      tokens: list[DependencyToken],
  ) -> bool:
    first = self.first_nonpunct_token(tokens)

    if first is None:
      return True

    if first.text.lower() not in self.WEAK_FRAGMENT_STARTS:
      return False

    return True

  def has_explicit_subject(
      self,
      tokens: list[DependencyToken],
  ) -> bool:
    return any(
      token.dependency_relation in {"nsubj", "nsubj:pass"}
      for token in tokens
    )

  def merge_strict_fragments(
      self,
      clauses: list[list[DependencyToken]],
  ) -> list[list[DependencyToken]]:
    merged = []

    for clause in clauses:
      if (
          self.should_merge_with_previous(clause)
          and merged
          and self.same_sentence_group(clause, merged[-1])
      ):
        merged[-1] = sorted(
          merged[-1] + clause,
          key=lambda token: token.global_token_id
        )
      else:
        merged.append(clause)

    return merged

  def same_sentence_group(
      self,
      tokens_a: list[DependencyToken],
      tokens_b: list[DependencyToken],
  ) -> bool:
    return self.sentence_ids(tokens_a) == self.sentence_ids(tokens_b)

  def sentence_ids(
      self,
      tokens: list[DependencyToken],
  ) -> set[int]:
    return {
      token.sentence_id
      for token in tokens
    }

  def is_infinitival_fragment(
      self,
      root: DependencyToken,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    if root.xpos == "VB":
      return any(
        token.text.lower() == "to" and token.dependency_relation == "mark"
        for token in subtree_tokens
      )

    return False

  def is_viable_relative_clause(
      self,
      subtree_tokens: list[DependencyToken]
  ) -> bool:
    nonpunct = self.nonpunct_tokens(subtree_tokens)
    content = self.content_tokens(subtree_tokens)

    has_subject = any(
      token.dependency_relation in {"nsubj", "nsubj:pass"}
      for token in subtree_tokens
    )

    has_content_beyond_relative_pronoun = any(
      token.upos in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
      and token.text.lower() not in {"who", "which", "that"}
      for token in subtree_tokens
    )

    return (
      has_subject
      and len(nonpunct) >= self.MIN_STRICT_NONPUNCT_TOKENS
      and len(content) >= self.MIN_STRICT_CONTENT_TOKENS
      and has_content_beyond_relative_pronoun
    )

  def filter_nested_strict_roots(
      self,
      split_roots: list[DependencyToken],
  ) -> list[DependencyToken]:
    filtered = []

    for root in split_roots:
      is_nested = False

      for other in split_roots:
        if root == other or root.sentence_id != other.sentence_id:
          continue

        other_subtree = self.collect_subtree(
          other.sentence_id,
          other.token_id
        )
        other_subtree_ids = {
          token.token_id
          for token in other_subtree
        }

        if root.token_id in other_subtree_ids:
          is_nested = True
          break

      if not is_nested:
        filtered.append(root)

    return filtered

  def is_descendent_of_any_split(
      self,
      token: DependencyToken,
      split_roots: list[DependencyToken]
  ) -> bool:
    for root in split_roots:
      descendents = self.collect_subtree(root.sentence_id, root.token_id)
      descendent_ids = {
        descendent.token_id
        for descendent in descendents
        if descendent.token_id != root.token_id
      }
      if token.token_id in descendent_ids:
        return True

    return False
