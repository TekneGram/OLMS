from OLMSClasses.O import O
from OLMSClasses.L import L
from OLMSClasses.M import M
from OLMSClasses.S import S
import pandas as pd
from TextProcessing.deprel import DependencyParse
from SLM.embedding import EmbeddingModel

class OLMSVector:
  def __init__(
      self,
      score_table: pd.DataFrame,
      parsed_1: DependencyParse,
      parsed_2: DependencyParse,
      response_1: str, 
      response_2: str,
      embedding_model: EmbeddingModel,
      ) -> None:
    
    self.organization = O(
      score_table=score_table,
    )

    self.lexis = L(
      text_a = response_1,
      text_b = response_2
    )

    self.meaning = M(
      response_1=response_1,
      response_2=response_2,
      embedding_model=embedding_model,
    )

    self.structure = S(
      parsed_a=parsed_1,
      parsed_b=parsed_2
    )

  def create_olms_vector(self) -> dict[str, float]:
    org_score = self.organization.soft_organization_score()
    lex_score = self.lexis.lexical_similarity()
    meaning_score = self.meaning.get_similarity_score()
    struct_score = self.structure.structure_score()

    return {
        "org": org_score,
        "lex": lex_score,
        "meaning": meaning_score,
        "struct": struct_score
      }
