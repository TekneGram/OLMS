# This script calculates a structural similarity score based on syntactic complexity.
from collections import Counter

import pandas as pd

from TextProcessing.deprel import DependencyParse, DependencyToken

class S:
  """
  Syntactic complexity similarity over whole responses.
  It calculates the following for each response using the parts of speech and dependency relations.
    - Subordinate clauses types per response:
      - advcl per response
      - acl per response
      - acl:relcl per response
      - ccomp per response
      - xcomp per response
      - csubj per response
    - Modifier density
      - adjective-noun per response: amod
      - adverb-verb/adjective/adverb per response: advmod
      - noun-noun per response: compound
      - nominal-postmodifers per response: nmod
      - appositive modification per response: appos
    - Predicate complexity
      - Passive constructions per response: nsubj:pass, aux:pass
      - core arguments per verb over the response: nsubj, obj,
    - Tree depth
      - average token depth across the response
      - number of embedded clausal nodes across the response
      - maximum dependency depth

  It uses these to create a feature vector for each response
  """

  SUBORDINATE_CLAUSE_DEPRELS = (
    "advcl",
    "acl",
    "acl:relcl",
    "ccomp",
    "xcomp",
    "csubj",
  )

  MODIFIER_DEPRELS = (
    "amod",
    "advmod",
    "compound",
    "nmod",
    "appos",
  )

  PREDICATE_DEPRELS = (
    "nsubj:pass",
    "aux:pass",
    "nsubj",
    "obj",
  )

  RATE_FEATURE_NAMES = (
    "advcl_per_sentence",
    "acl_per_sentence",
    "acl_relcl_per_sentence",
    "ccomp_per_sentence",
    "xcomp_per_sentence",
    "csubj_per_sentence",
    "amod_per_noun",
    "advmod_per_verb_adj_adv",
    "compound_per_noun",
    "nmod_per_noun",
    "appos_per_noun",
    "nsubj_pass_per_verb",
    "aux_pass_per_verb",
    "nsubj_per_verb",
    "obj_per_verb",
    "average_token_depth",
    "embedded_clausal_nodes_per_sentence",
    "maximum_dependency_depth",
    "mean_dependency_distance",
    "maximum_dependency_distance",
    "long_dependency_rate_ge_5",
  )

  TOTAL_FEATURE_NAMES = (
    "advcl",
    "acl",
    "acl_relcl",
    "ccomp",
    "xcomp",
    "csubj",
    "amod",
    "advmod",
    "compound",
    "nmod",
    "appos",
    "nsubj_pass",
    "aux_pass",
    "nsubj",
    "obj",
    "total_token_depth",
    "embedded_clausal_nodes",
    "maximum_dependency_depth",
    "total_dependency_distance",
    "maximum_dependency_distance",
    "long_dependencies_ge_5",
  )

  FEATURE_NAMES = RATE_FEATURE_NAMES

  def __init__(
      self,
      parsed_a: DependencyParse,
      parsed_b: DependencyParse,
      skip_punct: bool = True,
      long_dependency_threshold: int = 5,
  ) -> None:
    self.parsed_a = parsed_a
    self.parsed_b = parsed_b
    self.skip_punct = skip_punct
    self.long_dependency_threshold = long_dependency_threshold

  def structure_score(self) -> float:
    """
    Return syntactic complexity similarity over whole responses.
    The final score is the mean of normalized-rate similarity and raw-total
    similarity.
    """
    normalized_score = self.normalized_structure_score()
    total_score = self.total_structure_score()

    if normalized_score == 0.0 and total_score == 0.0:
      return 0.0

    return float((normalized_score + total_score) / 2.0)

  def normalized_structure_score(self) -> float:
    """
    Return feature-wise similarity over normalized syntactic complexity rates.
    """
    table = self.feature_table()

    if table.empty:
      return 0.0

    return float(table["similarity"].mean())

  def total_structure_score(self) -> float:
    """
    Return feature-wise similarity over raw syntactic complexity totals.
    """
    table = self.total_feature_table()

    if table.empty:
      return 0.0

    return float(table["similarity"].mean())

  def score_breakdown(self) -> dict[str, float]:
    """
    Return the normalized, total, and combined structure scores.
    """
    normalized_score = self.normalized_structure_score()
    total_score = self.total_structure_score()

    return {
      "normalized_struct": normalized_score,
      "total_struct": total_score,
      "struct": float((normalized_score + total_score) / 2.0),
    }

  def feature_table(self) -> pd.DataFrame:
    """
    Return diagnostics for each normalized syntactic complexity feature.
    """
    rates_a = self.feature_rates(self.parsed_a)
    rates_b = self.feature_rates(self.parsed_b)

    rows = []
    for feature_name in self.RATE_FEATURE_NAMES:
      value_a = rates_a[feature_name]
      value_b = rates_b[feature_name]
      rows.append({
        "feature": feature_name,
        "a_value": value_a,
        "b_value": value_b,
        "similarity": self._feature_similarity(value_a, value_b),
      })

    return pd.DataFrame(
      rows,
      columns=[
        "feature",
        "a_value",
        "b_value",
        "similarity",
      ]
    )

  def total_feature_table(self) -> pd.DataFrame:
    """
    Return diagnostics for each raw syntactic complexity total.
    """
    totals_a = self.feature_totals(self.parsed_a)
    totals_b = self.feature_totals(self.parsed_b)

    rows = []
    for feature_name in self.TOTAL_FEATURE_NAMES:
      value_a = totals_a[feature_name]
      value_b = totals_b[feature_name]
      rows.append({
        "feature": feature_name,
        "a_value": value_a,
        "b_value": value_b,
        "similarity": self._feature_similarity(value_a, value_b),
      })

    return pd.DataFrame(
      rows,
      columns=[
        "feature",
        "a_value",
        "b_value",
        "similarity",
      ]
    )

  def feature_rates(
      self,
      parsed: DependencyParse,
  ) -> dict[str, float]:
    tokens = self._content_tokens(parsed)
    token_by_sentence_and_id = {
      (token.sentence_id, token.token_id): token
      for token in tokens
    }

    n_sentences = len({token.sentence_id for token in tokens})
    n_nouns = sum(
      token.upos in {"NOUN", "PROPN", "PRON"}
      for token in tokens
    )
    n_verbs = sum(
      token.upos in {"VERB", "AUX"}
      for token in tokens
    )
    n_eligible_advmod_heads = sum(
      token.upos in {"VERB", "ADJ", "ADV"}
      for token in tokens
    )

    deprel_counts = Counter(
      token.dependency_relation
      for token in tokens
    )

    depths = self._dependency_depths(tokens, token_by_sentence_and_id)
    distances = self._dependency_distances(tokens)
    embedded_clausal_nodes = sum(
      deprel_counts[deprel]
      for deprel in self.SUBORDINATE_CLAUSE_DEPRELS
    )

    rates = {
      "advcl_per_sentence": self._safe_divide(deprel_counts["advcl"], n_sentences),
      "acl_per_sentence": self._safe_divide(deprel_counts["acl"], n_sentences),
      "acl_relcl_per_sentence": self._safe_divide(deprel_counts["acl:relcl"], n_sentences),
      "ccomp_per_sentence": self._safe_divide(deprel_counts["ccomp"], n_sentences),
      "xcomp_per_sentence": self._safe_divide(deprel_counts["xcomp"], n_sentences),
      "csubj_per_sentence": self._safe_divide(deprel_counts["csubj"], n_sentences),
      "amod_per_noun": self._safe_divide(deprel_counts["amod"], n_nouns),
      "advmod_per_verb_adj_adv": self._safe_divide(deprel_counts["advmod"], n_eligible_advmod_heads),
      "compound_per_noun": self._safe_divide(deprel_counts["compound"], n_nouns),
      "nmod_per_noun": self._safe_divide(deprel_counts["nmod"], n_nouns),
      "appos_per_noun": self._safe_divide(deprel_counts["appos"], n_nouns),
      "nsubj_pass_per_verb": self._safe_divide(deprel_counts["nsubj:pass"], n_verbs),
      "aux_pass_per_verb": self._safe_divide(deprel_counts["aux:pass"], n_verbs),
      "nsubj_per_verb": self._safe_divide(deprel_counts["nsubj"], n_verbs),
      "obj_per_verb": self._safe_divide(deprel_counts["obj"], n_verbs),
      "average_token_depth": self._mean(depths),
      "embedded_clausal_nodes_per_sentence": self._safe_divide(embedded_clausal_nodes, n_sentences),
      "maximum_dependency_depth": float(max(depths, default=0)),
      "mean_dependency_distance": self._mean(distances),
      "maximum_dependency_distance": float(max(distances, default=0)),
      "long_dependency_rate_ge_5": self._safe_divide(
        sum(
          distance >= self.long_dependency_threshold
          for distance in distances
        ),
        len(distances),
      ),
    }

    return rates

  def feature_totals(
      self,
      parsed: DependencyParse,
  ) -> dict[str, float]:
    tokens = self._content_tokens(parsed)
    token_by_sentence_and_id = {
      (token.sentence_id, token.token_id): token
      for token in tokens
    }

    deprel_counts = Counter(
      token.dependency_relation
      for token in tokens
    )

    depths = self._dependency_depths(tokens, token_by_sentence_and_id)
    distances = self._dependency_distances(tokens)
    embedded_clausal_nodes = sum(
      deprel_counts[deprel]
      for deprel in self.SUBORDINATE_CLAUSE_DEPRELS
    )

    return {
      "advcl": float(deprel_counts["advcl"]),
      "acl": float(deprel_counts["acl"]),
      "acl_relcl": float(deprel_counts["acl:relcl"]),
      "ccomp": float(deprel_counts["ccomp"]),
      "xcomp": float(deprel_counts["xcomp"]),
      "csubj": float(deprel_counts["csubj"]),
      "amod": float(deprel_counts["amod"]),
      "advmod": float(deprel_counts["advmod"]),
      "compound": float(deprel_counts["compound"]),
      "nmod": float(deprel_counts["nmod"]),
      "appos": float(deprel_counts["appos"]),
      "nsubj_pass": float(deprel_counts["nsubj:pass"]),
      "aux_pass": float(deprel_counts["aux:pass"]),
      "nsubj": float(deprel_counts["nsubj"]),
      "obj": float(deprel_counts["obj"]),
      "total_token_depth": float(sum(depths)),
      "embedded_clausal_nodes": float(embedded_clausal_nodes),
      "maximum_dependency_depth": float(max(depths, default=0)),
      "total_dependency_distance": float(sum(distances)),
      "maximum_dependency_distance": float(max(distances, default=0)),
      "long_dependencies_ge_5": float(
        sum(
          distance >= self.long_dependency_threshold
          for distance in distances
        )
      ),
    }

  def _content_tokens(
      self,
      parsed: DependencyParse,
  ) -> list[DependencyToken]:
    if not self.skip_punct:
      return list(parsed.tokens)

    return [
      token
      for token in parsed.tokens
      if token.upos != "PUNCT"
    ]

  def _dependency_depths(
      self,
      tokens: list[DependencyToken],
      token_by_sentence_and_id: dict[tuple[int, int], DependencyToken],
  ) -> list[int]:
    depths = []

    for token in tokens:
      depth = 0
      current = token
      seen = set()

      while (
          current.head_token_id is not None
          and (current.sentence_id, current.token_id) not in seen
      ):
        seen.add((current.sentence_id, current.token_id))
        depth += 1

        head = token_by_sentence_and_id.get(
          (current.sentence_id, current.head_token_id)
        )
        if head is None:
          break

        current = head

      depths.append(depth)

    return depths

  def _dependency_distances(
      self,
      tokens: list[DependencyToken],
  ) -> list[int]:
    return [
      abs(token.token_id - token.head_token_id)
      for token in tokens
      if token.head_token_id is not None
    ]

  def _feature_similarity(
      self,
      value_a: float,
      value_b: float,
  ) -> float:
    if value_a == 0.0 and value_b == 0.0:
      return 1.0

    denominator = max(abs(value_a), abs(value_b))
    if denominator == 0.0:
      return 0.0

    return 1.0 - abs(value_a - value_b) / denominator

  def _safe_divide(
      self,
      numerator: float,
      denominator: float,
  ) -> float:
    if denominator == 0:
      return 0.0

    return float(numerator / denominator)

  def _mean(
      self,
      values: list[int],
  ) -> float:
    if not values:
      return 0.0

    return float(sum(values) / len(values))


##################


class S_old:
  """
  Structural similarity over semantically matched clause pairs.

  Implements:

    Struct(D_a, D_b) = (1 / |R|) * sum J(P(c_i), P(c_j))

  where R is the set of matched clause pairs returned by PairMatcher.match(),
  P(c) is a multiset of syntactic patterns from clause c, and J is multiset
  Jaccard similarity.
  """

  REQUIRED_COLUMNS = {
    "clause_in_a_index",
    "clause_in_b_index",
  }

  def __init__(
      self,
      match_table: pd.DataFrame,
      clauses_a_tokens: list[list[DependencyToken]],
      clauses_b_tokens: list[list[DependencyToken]],
      skip_punct: bool = True,
  ) -> None:
    missing_columns = self.REQUIRED_COLUMNS - set(match_table.columns)
    if missing_columns:
      raise ValueError(
        f"The paired clauses table from pair_matcher.py is missing required columns: {sorted(missing_columns)}"
      )

    self.match_table = match_table.copy()
    self.clauses_a_tokens = clauses_a_tokens
    self.clauses_b_tokens = clauses_b_tokens
    self.skip_punct = skip_punct

    self._validate_clause_indices()
    return

  def structure_score(self) -> float:
    """
    Return the average multiset Jaccard similarity over matched clause pairs.
    If there are no matched pairs, return 0.0.
    """
    if self.match_table.empty:
      return 0.0

    scores = self.pair_scores()
    return float(scores["structurality"].mean())

  def pair_scores(self) -> pd.DataFrame:
    """
    Return structurality diagnostics for each matched clause pair.
    """
    rows = []

    for _, row in self.match_table.iterrows():
      clause_in_a_index = int(row["clause_in_a_index"])
      clause_in_b_index = int(row["clause_in_b_index"])

      patterns_a = self._syntactic_patterns(
        self.clauses_a_tokens[clause_in_a_index]
      )
      patterns_b = self._syntactic_patterns(
        self.clauses_b_tokens[clause_in_b_index]
      )

      rows.append({
        "clause_in_a_index": clause_in_a_index,
        "clause_in_b_index": clause_in_b_index,
        "structurality": self._multiset_jaccard(patterns_a, patterns_b),
        "n_patterns_a": int(patterns_a.total()),
        "n_patterns_b": int(patterns_b.total()),
        "n_pattern_types_a": len(patterns_a),
        "n_pattern_types_b": len(patterns_b),
        "n_shared_pattern_types": len(set(patterns_a) & set(patterns_b)),
      })

    return pd.DataFrame(
      rows,
      columns=[
        "clause_in_a_index",
        "clause_in_b_index",
        "structurality",
        "n_patterns_a",
        "n_patterns_b",
        "n_pattern_types_a",
        "n_pattern_types_b",
        "n_shared_pattern_types",
      ]
    )

  def _syntactic_patterns(
      self,
      clause_tokens: list[DependencyToken],
  ) -> Counter[tuple[str, str, str]]:
    """
    Build the multiset of <pos_t, dep_r, pos_h> patterns for one clause.
    Only include dependency edges where both token and head are inside the same
    clause.
    """
    token_by_id = {
      token.token_id: token
      for token in clause_tokens
    }

    patterns: Counter[tuple[str, str, str]] = Counter()

    for token in clause_tokens:
      if self.skip_punct and token.upos == "PUNCT":
        continue

      if token.head_token_id is None:
        continue

      head = token_by_id.get(token.head_token_id)
      if head is None:
        continue

      if self.skip_punct and head.upos == "PUNCT":
        continue

      patterns[
        (
          token.upos,
          token.dependency_relation,
          head.upos,
        )
      ] += 1

    return patterns

  def _multiset_jaccard(
      self,
      patterns_a: Counter[tuple[str, str, str]],
      patterns_b: Counter[tuple[str, str, str]],
  ) -> float:
    keys = set(patterns_a) | set(patterns_b)

    if not keys:
      return 0.0

    numerator = sum(
      min(patterns_a[key], patterns_b[key])
      for key in keys
    )

    denominator = sum(
      max(patterns_a[key], patterns_b[key])
      for key in keys
    )

    if denominator == 0:
      return 0.0

    return numerator / denominator

  def _validate_clause_indices(self) -> None:
    if self.match_table.empty:
      return

    max_a = len(self.clauses_a_tokens) - 1
    max_b = len(self.clauses_b_tokens) - 1

    bad_a = sorted(
      {
        int(index)
        for index in self.match_table["clause_in_a_index"]
        if int(index) < 0 or int(index) > max_a
      }
    )

    bad_b = sorted(
      {
        int(index)
        for index in self.match_table["clause_in_b_index"]
        if int(index) < 0 or int(index) > max_b
      }
    )

    errors = []
    if bad_a:
      errors.append(
        f"A-side clause indices out of range 0..{max_a}: {bad_a}"
      )

    if bad_b:
      errors.append(
        f"B-side clause indices out of range 0..{max_b}: {bad_b}"
      )

    if errors:
      raise ValueError("; ".join(errors))
