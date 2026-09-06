from TextProcessing.deprel import DependencyParser
from TextProcessing.clause_splitter import ClauseSplitter
from TextProcessing.bertscorer import BertScore
from TextProcessing.pair_matcher import PairMatcher
from OLMSClasses.O import O
from OLMSClasses.L import L
from OLMSClasses.M import M
from OLMSClasses.S import S

def main() -> None:
  my_text = "The thesis statement clearly presents your opinion, but it needs to explain why the topic matters. Although your thesis statement is focused, it does not fully preview the reasons you discuss in the essay. Your thesis says that school uniforms are good, but it should state your exact position more clearly. To make the thesis stronger, add a reason that connects your opinion to the main argument. Your thesis is understandable because it gives the reader a clear opinion. The thesis would be clearer if you named the two reasons that your body paragraph will explain. You have a thesis statement, but because it is very general, the reader may not know your main argument. Your thesis statement, which appears at the end of the introduction, gives an opinion but does not include a clear reason. I suggest revising the thesis so that it includes both your position and the reason for that position. The thesis is not specific enough to guide the rest of the essay."
  my_text_2 = "The thesis statement clearly presents your opinion. However, it needs to explain why the topic matters. Your thesis statement is focused. However, it does not fully preview the reasons you discuss in the essay. Your thesis says that school uniforms are good. But it should state your exact position more clearly. To make the thesis stronger add a reason. The reason should connect your opinion to the main argument. Your thesis is understandable. This is because it gives the reader a clear opinion. The thesis could be clearer. Name the two reasons that your body paragraph will explain. You have a thesis statement. But it is very general. So the reader may not know your main argument. Your thesis statement gives an opinion. This appears at the end of the introduction. However, it does not include a clear reason. I suggest revising the thesis. It should include both your position and the reason for that position. The thesis is not specific enough to guide the rest of the essay."

  # Add a text length checker.
  # If the text length is less than 100 words for any text, then we cannot calculate the MTLD score in L.py

  deprel = DependencyParser(
    ".models/udpipe"
  )

  parsed_1 = deprel.parse("trial", my_text)
  parsed_2 = deprel.parse("trial_2", my_text_2)
  splitter_1 = ClauseSplitter(parsed_1)
  splitter_2 = ClauseSplitter(parsed_2)

  clauses_1_tokens = splitter_1.split_all_tokens()
  clauses_2_tokens = splitter_2.split_all_tokens()

  clauses_1 = [splitter_1.token_text(clause) for clause in clauses_1_tokens]
  clauses_2 = [splitter_2.token_text(clause) for clause in clauses_2_tokens]

  # for i, clause in enumerate(clauses_1):
  #   print(i, clause)

  # for i, clause_tokens in enumerate(clauses_1_tokens):
  #   for token in sorted(clause_tokens, key=lambda token: token.token_id):
  #     print(
  #       token.token_id,
  #       token.text,
  #       token.upos,
  #       token.dependency_relation,
  #       token.head_token_id
  #     )

  bertScorer = BertScore(clauses_1, clauses_2)
  bert_score_table = bertScorer.create_bert_score_table()

  # Setting min_score = 0.35 appears to be more inclusive
  # Setting min_score = 0.3 leads to clause matches that seem a bit weird
  # Setting min_score = 0.5 possibly drops some meaningful matches, but OLMS scores are higher
  matcher = PairMatcher(
    score_table=bert_score_table,
    min_score=0.50,
    metric="f1",
    capacity=2
  )

  clause_pairs = matcher.match()
  print(clause_pairs)

  organization = O(
    match_table=clause_pairs,
    n_clauses_a=clause_pairs["clause_in_a_index"].max(),
    n_clauses_b=clause_pairs["clause_in_b_index"].max()
  )

  organization_score = organization.organization_score()
  print(organization_score)

  meaning = M(
    match_table=clause_pairs
  )
  semantics_score = meaning.semantics_score()
  print(semantics_score)

  lexis = L(
    text_a = my_text,
    text_b = my_text_2
  )
  # In real implementation, input min of text length of my_text and my_text_2 for text_size
  # This will lead to the lexical similarity choosing between mtld and mattr (not sure if this approach is valid though)
  lexical_similarity_score = lexis.lexical_similarity(text_size=101)
  print(lexical_similarity_score)

  structure = S(
    match_table=clause_pairs,
    clauses_a_tokens=clauses_1_tokens,
    clauses_b_tokens=clauses_2_tokens
  )
  structural_similarity_score = structure.structure_score()
  print(structural_similarity_score)





  return

if __name__ == "__main__":
  main()
