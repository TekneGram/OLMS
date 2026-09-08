from TextProcessing.deprel import DependencyParser
from TextProcessing.clause_splitter import ClauseSplitter
from TextProcessing.bertscorer import BertScore
from TextProcessing.tf_idf_scorer import TFIDFScore
from TextProcessing.pair_matcher import PairMatcher
from TextProcessing.sentence_splitter import SentenceSplitter
from OLMSClasses.O import O
from OLMSClasses.L import L
from OLMSClasses.M import M
from OLMSClasses.S import S

from OLMSClasses.OLMS_vector import OLMSVector

import configuration
import pandas as pd
from pathlib import Path
from SLM.essay_feedback import EssayFeedback

from SLM.slm import SLM

def main() -> None:

  # Set up file names and data saving paths.
  data_name = "feedback_data"
  data_dir = Path("data")
  data_dir.mkdir(exist_ok=True)

  clauses_path = data_dir / f"{data_name}_clauses.md"
  clause_pairs_path = data_dir / f"{data_name}_clause_pairs.md"
  structure_path = data_dir / f"{data_name}_structure.md"
  olms_vectors_path = data_dir / f"{data_name}_olms_vectors.csv"

  clauses_path.write_text("", encoding="utf-8")
  clause_pairs_path.write_text("", encoding="utf-8")
  structure_path.write_text("", encoding="utf-8")

  # Language model - generate feedback on essays.
  fb = EssayFeedback()
  all_feedback = fb.generate_multiple_feedback(output_filename=data_name)

  # Prepare dependency parser and object to accept results
  deprel = DependencyParser(".models/udpipe")
  olms_vectors = []

  # Loop through all the feedback generated
  for feedback in all_feedback:
    essay_file = feedback["essay_file"]
    responses = feedback["responses"]

    if len(responses) < 2:
      raise ValueError(f"Expected at least 2 responses for {essay_file}")

    # Retrieve language model responses
    response_1 = responses[0]
    response_2 = responses[1]

    # Parse them for parts of speech and dependency relations with udpipe
    parsed_1 = deprel.parse(f"{essay_file}_response_1", response_1)
    parsed_2 = deprel.parse(f"{essay_file}_response_2", response_2)

    # Split responses into units (clauses or sentences)
    splitter_1 = SentenceSplitter(parsed_1)
    splitter_2 = SentenceSplitter(parsed_2)
    clauses_1_tokens = splitter_1.split_all_tokens()
    clauses_2_tokens = splitter_2.split_all_tokens()
    clauses_1 = [splitter_1.token_text(clause) for clause in clauses_1_tokens]
    clauses_2 = [splitter_2.token_text(clause) for clause in clauses_2_tokens]
    SentenceSplitter.append_clause_sets_to_markdown(
      clauses_path,
      essay_file,
      {
        "Response A Clauses": clauses_1,
        "Response B Clauses": clauses_2
      }
    )

    # Create the BERTScore matrix for all units between responses
    bertScorer = BertScore(clauses_1, clauses_2)
    bert_score_table = bertScorer.create_bert_score_table()
    tfidfScorer = TFIDFScore(clauses_1, clauses_2)
    tfidf_score_table = tfidfScorer.create_tfidf_score_table()

    # With new Organization measure, we likely do *not* need a pair matcher
    # anymore.
    # matcher = PairMatcher(
    #   score_table=bert_score_table,
    #   tfidf_score_table=tfidf_score_table,
    #   min_score=configuration.min_score,
    #   metric="f1",
    #   capacity=configuration.capacity,
    #   use_position_bias=True,
    #   position_weight=0.15,
    # )
    # clause_pairs = matcher.match()
    # matcher.append_matches_to_markdown(
    #   clause_pairs_path,
    #   essay_file,
    #   clause_pairs
    # )
    # Loop through all essay feedback

    # Run OLMS measures and generate the OLMS vector for these two responses.
    olms_vector = OLMSVector(
      score_table=bert_score_table,
      parsed_1=parsed_1,
      parsed_2=parsed_2,
      response_1=response_1,
      response_2=response_2
    )
    vector = olms_vector.create_olms_vector()
    olms_vector.structure.append_structure_to_markdown(
      structure_path,
      essay_file,
    )
    vector["essay_file"] = essay_file
    olms_vectors.append(vector)

  pd.DataFrame(olms_vectors).to_csv(olms_vectors_path, index=False)
  print(olms_vectors)

  # CHECKS HERE

  my_text = "The thesis statement in the introduction is Although wearable fitness devices can help students become more aware of their sleep and exercise habits this essay argues that they should be used carefully because their benefits are limited by weak long term behavior change privacy concerns and possible stress. This statement has a main idea because it clearly states the essay 's main point about wearable devices and why they should be used carefully. It also has an opinion because the writer is taking a stance that these devices have limitations and should be used carefully. The statement previews the content of the essay by mentioning the three main reasons for caution. long term behavior change privacy concerns and stress. The statement is not too complex and is written in clear simple language. The thesis statement is clear and directly states the writer 's main argument. It is well structured and covers the key points the essay will discuss. The writer could improve by making the statement slightly more concise but it is overall a good thesis statement."
  my_text_2 = "The thesis statement in the introduction is Although wearable fitness devices can help students become more aware of their sleep and exercise habits this essay argues that they should be used carefully because their benefits are limited by weak long term behavior change privacy concerns and possible stress. The thesis statement has a main idea because it states the overall point of the essay that wearable devices should be used carefully. It also expresses an opinion by saying they should be used carefully. The statement previews the content of the essay by mentioning the reasons why they should be used carefully. The statement is not too complex and is written in simple sentences. However it could be more specific. For example it could mention that the essay will discuss how wearable devices can help students but also have problems. The statement is clear and direct but it could be improved to be more concise. The writer should make sure the thesis statement clearly states the main argument and includes the key points that will be discussed in the essay."

  deprel = DependencyParser(
    ".models/udpipe"
  )

  parsed_1 = deprel.parse("trial", my_text)
  parsed_2 = deprel.parse("trial_2", my_text_2)
  splitter_1 = SentenceSplitter(parsed_1)
  splitter_2 = SentenceSplitter(parsed_2)

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
  tfidfScorer = TFIDFScore(clauses_1, clauses_2)
  tfidf_score_table = tfidfScorer.create_tfidf_score_table()

  # Setting min_score = 0.35 appears to be more inclusive
  # Setting min_score = 0.3 leads to clause matches that seem a bit weird
  # Setting min_score = 0.5 possibly drops some meaningful matches, but OLMS scores are higher
  # matcher = PairMatcher(
  #   score_table=bert_score_table,
  #   tfidf_score_table=tfidf_score_table,
  #   min_score=configuration.min_score,
  #   metric="f1",
  #   capacity=configuration.capacity
  # )

  # clause_pairs = matcher.match()
  # print(clause_pairs)

  organization = O(
    score_table=bert_score_table,
  )

  organization_score = organization.organization_score()
  print("O Score: ", organization_score)

  lexis = L(
      text_a = my_text,
      text_b = my_text_2
    )
  # In real implementation, input min of text length of my_text and my_text_2 for text_size
  # This will lead to the lexical similarity choosing between mtld and mattr (not sure if this approach is valid though)
  lexical_similarity_score = lexis.lexical_similarity(text_size=101)
  print("L Score: ", lexical_similarity_score)

  meaning = M(
    response_1=my_text,
    response_2=my_text_2
  )
  semantics_score = meaning.meaning_similarity()
  print("M Score: ", semantics_score)

  

  structure = S(
    parsed_a=parsed_1,
    parsed_b=parsed_2
  )
  structural_similarity_score = structure.structure_score()
  print("S Score: ", structural_similarity_score)





  return

if __name__ == "__main__":
  main()
