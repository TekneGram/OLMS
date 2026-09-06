from deprel import DependencyParser
from clause_splitter import ClauseSplitter
from bertscorer import BertScore
from pair_matcher import PairMatcher
from O import O
def main() -> None:
  my_text = "The thesis statement clearly presents your opinion, but it needs to explain why the topic matters. Although your thesis statement is focused, it does not fully preview the reasons you discuss in the essay. Your thesis says that school uniforms are good, but it should state your exact position more clearly. To make the thesis stronger, add a reason that connects your opinion to the main argument. Your thesis is understandable because it gives the reader a clear opinion. The thesis would be clearer if you named the two reasons that your body paragraph will explain. You have a thesis statement, but because it is very general, the reader may not know your main argument. Your thesis statement, which appears at the end of the introduction, gives an opinion but does not include a clear reason. I suggest revising the thesis so that it includes both your position and the reason for that position. The thesis is not specific enough to guide the rest of the essay."
  my_text_2 = "The thesis statement clearly presents your opinion. However, it needs to explain why the topic matters. Your thesis statement is focused. However, it does not fully preview the reasons you discuss in the essay. Your thesis says that school uniforms are good. But it should state your exact position more clearly. To make the thesis stronger add a reason. The reason should connect your opinion to the main argument. Your thesis is understandable. This is because it gives the reader a clear opinion. The thesis could be clearer. Name the two reasons that your body paragraph will explain. You have a thesis statement. But it is very general. So the reader may not know your main argument. Your thesis statement gives an opinion. This appears at the end of the introduction. However, it does not include a clear reason. I suggest revising the thesis. It should include both your position and the reason for that position. The thesis is not specific enough to guide the rest of the essay."

  deprel = DependencyParser(
    ".models/udpipe"
  )

  parsed_1 = deprel.parse("trial", my_text)
  parsed_2 = deprel.parse("trial_2", my_text_2)
  splitter_1 = ClauseSplitter(parsed_1)
  splitter_2 = ClauseSplitter(parsed_2)
  clauses_1 = splitter_1.split_all()
  clauses_2 = splitter_2.split_all()

  bertScorer = BertScore(clauses_1, clauses_2)
  bert_score_table = bertScorer.create_bert_score_table()

  matcher = PairMatcher(
    score_table=bert_score_table,
    min_score=0.20,
    metric="f1",
    capacity=2
  )

  clause_pairs = matcher.match()
  organization = O(
    match_table=clause_pairs
  )

  tau = organization.normalized_tie_aware_tau()
  print(tau)




  return

if __name__ == "__main__":
  main()
