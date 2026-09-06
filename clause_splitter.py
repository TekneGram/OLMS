from deprel import DependencyToken, DependencyParse

class ClauseSplitter:
  SPLIT_DEPRELS = {"conj", "advcl", "acl:relcl"}
  CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

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

  def split_sentence(
      self,
      sentence_tokens: list[DependencyToken],
  ) -> list[str]:
    return [
      self.token_text(clause_tokens)
      for clause_tokens in self.split_sentence_tokens(sentence_tokens)
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
    tokens = sorted(tokens, key=lambda token: token.token_id)
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
