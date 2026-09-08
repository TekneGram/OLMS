from OLMSClasses.O import O
from OLMSClasses.L import L
from OLMSClasses.M import M
from OLMSClasses.S import S
import pandas as pd
from TextProcessing.deprel import DependencyToken

class OLMSVector:
  def __init__(
      self,
      score_table:pd.DataFrame, 
      clause_pairs: pd.DataFrame, 
      clauses_a: list[str],
      clauses_b: list[str],
      clauses_tokens_a: list[list[DependencyToken]],
      clauses_tokens_b: list[list[DependencyToken]],
      response_1: str, 
      response_2: str
      ) -> None:
    
    self.organization = O(
      score_table=score_table,
    )

    self.lexis = L(
      text_a = response_1,
      text_b = response_2
    )

    self.meaning = M(
      match_table=clause_pairs
    )

    self.structure = S(
      match_table=clause_pairs,
      clauses_a_tokens=clauses_tokens_a,
      clauses_b_tokens = clauses_tokens_b
    )

  def create_olms_vector(self) -> dict[str, float]:
    org_score = self.organization.organization_score()
    lex_score = self.lexis.lexical_similarity()
    meaning_score = self.meaning.semantics_score()
    struct_score = self.structure.structure_score()

    return {
        "org": org_score,
        "lex": lex_score,
        "meaning": meaning_score,
        "struct": struct_score
      }