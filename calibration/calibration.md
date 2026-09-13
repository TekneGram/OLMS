# Calibration Commentary

## Overview

The calibration texts are controlled paraphrase and reorganization examples. Sections I and II are close paraphrases with broadly matching paragraph and sentence progression. Section III expresses the same content in shorter, simpler sentences. Section IV retains much of the content of I and II but reorganizes it, often beginning with a conclusion or recommendation and then moving backwards through supporting details.

Across the four topics, the mean scores by comparison are:

| Comparison | O | L | M | S |
|---|---:|---:|---:|---:|
| I-II | 0.994 | 0.707 | 0.859 | 0.813 |
| I-III | 0.874 | 0.611 | 0.720 | 0.637 |
| II-III | 0.873 | 0.614 | 0.732 | 0.657 |
| I-IV | 0.514 | 0.870 | 0.831 | 0.799 |
| II-IV | 0.532 | 0.688 | 0.768 | 0.820 |
| III-IV | 0.660 | 0.606 | 0.675 | 0.636 |

The results suggest that the components are responding to different properties rather than producing interchangeable scores. I-II is the closest comparison overall in organization, meaning, and structure. I-IV is much less similar in organization but relatively close in lexis, meaning, and structure. This is the central calibration result: IV appears to preserve content and syntactic style while changing the order of presentation.

## O: Organization

`O` measures the similarity of the ordering of sentence or clause material. It uses the BERTScore similarity matrix, applies a low-temperature soft alignment, compares normalized positions, and averages the two directions. It is therefore an order-alignment measure, not a direct assessment of whether an argument is logically good or coherent.

The I-II scores are extremely high for every text: 0.989–0.998, with a mean of 0.994. This is expected because I and II preserve almost exactly the same progression of ideas: introduction of the topic, benefits, limitations, recommendations, and conclusion.

The I-III and II-III scores remain fairly high, around 0.85–0.88. Section III changes the sentence segmentation and simplifies the prose, but it generally maintains the same progression of ideas. This indicates that O is relatively tolerant of changes in sentence length when the discourse order remains stable.

The I-IV and II-IV scores are much lower, with means of 0.514 and 0.532. This is the strongest effect in the calibration. Section IV commonly begins with material that appears near the end of I or II, then moves through recommendations and supporting reasons in a different order. O detects this reordering even when the same topic and propositions are retained. The lower scores should therefore be interpreted as evidence of changed organization, not necessarily changed meaning or poor writing.

The III-IV scores are higher than the corresponding I-IV and II-IV scores in A, B, and C, although still moderate. This may indicate that the shorter III texts and reorganized IV texts share less rigid sentence-by-sentence alignment, making their relative ordering less different than the more linearly organized I and II texts. The pattern is consistent with O measuring positional alignment rather than content similarity.

## L: Lexical and phraseological similarity

`L` currently combines word-level TF-IDF similarity with either MTLD or MATTR similarity. With the default `text_size=99`, the current implementation uses MATTR. It then adds the new phraseology score, based on smoothed Jensen–Shannon similarity of sentence-bounded 2-gram and 3-gram distributions, using:

\[
L = 0.8L_{current} + 0.2S_{phraseology}
\]

The I-II scores are moderate rather than extremely high: 0.676–0.723. Although I and II are close paraphrases, they deliberately replace many words and expressions. TF-IDF and the 2-/3-gram JSD component therefore detect substantial surface-form difference. This is useful evidence that L is measuring lexical and phraseological form, rather than merely topic retention.

Section III receives lower L scores against I and II, with means of 0.611 and 0.614. This reflects both vocabulary simplification and shorter, less elaborate phrasing. For example, complex expressions in I and II are often replaced by short clauses in III. The result is consistent with L being sensitive to lexical choice and phrase distribution.

The I-IV scores are unexpectedly high: 0.823–0.936, with a mean of 0.870. This is explained by IV retaining many of the original lexical expressions and longer academic phrase patterns from I, even though the order is changed. The C comparison is especially high at 0.936, where IV closely reuses wording from I. This demonstrates that L is largely independent of organization: rearranging similar phrases does not substantially reduce lexical similarity.

II-IV scores are lower than I-IV in all four texts except that they remain above the III-IV scores. This suggests that IV is generally closer in wording to I than to II, despite being conceptually related to both. The lower III-IV scores confirm that simpler wording and altered phrase distributions are being detected.

The calibration does not expose the separate JSD sub-scores in the results report, so it is not possible to determine how much of each L score comes specifically from 2-grams versus 3-grams. A future diagnostic report should include those values. In particular, phraseology may be contributing to the high I-IV scores because IV deliberately reuses multi-word expressions from the earlier versions.

## M: Meaning

`M` averages two measures over the complete responses:

1. BERTScore F1, which compares token-level contextual similarity.
2. Embedding cosine similarity transformed from `[-1, 1]` to `[0, 1]`.

The I-II scores are consistently high, from 0.852 to 0.876, showing that the close paraphrases preserve their meaning well despite lexical substitutions.

The I-III and II-III scores fall to approximately 0.70–0.75. Section III still communicates the same main ideas, but it compresses, simplifies, and sometimes slightly generalizes the content. M therefore treats III as semantically related but less closely equivalent than I and II.

The I-IV scores are also high, averaging 0.831. This is important because O is low for the same comparisons. The combination of low O and high M indicates that IV changes the order of propositions while preserving much of their semantic content. M is consequently doing what it is intended to do: measuring whole-response meaning similarity rather than discourse order.

The II-IV mean is lower at 0.768, and III-IV is lower again at 0.675. These results suggest that IV retains more of the detailed content and wording of I and II than of the simplified III version. M is not simply responding to shared topic; it is sensitive to the amount of propositional detail retained.

M should not be interpreted as a score for factual accuracy, argument quality, or comprehensibility. It measures similarity between the two responses. A high M means that the responses express similar content, not that either response is correct or well written independently.

## S: Structural complexity

`S` compares syntactic-complexity profiles derived from dependency parses. It includes subordinate-clause types, modifier density, predicate-related features, dependency depth, dependency distance, and related counts. The final score averages similarity over normalized rates and raw totals.

The I-II scores are high overall, with a mean of 0.813. B and C are particularly high at 0.850 and 0.858, while A is lower at 0.715. This shows that close paraphrases usually preserve a similar syntactic register, but lexical substitution can still change dependency features enough to reduce S for an individual text.

The I-III and II-III scores are lower, averaging 0.637 and 0.657. This is expected from the calibration design: III breaks long sentences into shorter ones and uses simpler constructions. The lower S scores therefore provide evidence that the measure is detecting syntactic simplification rather than merely counting words.

The I-IV and II-IV scores are relatively high, averaging 0.799 and 0.820. Reordering the sentences does not necessarily change the dependency structures within those sentences, and IV retains the longer syntactic style of I and II. This contrasts with O: S is primarily sensitive to the internal complexity of the sentences, whereas O is sensitive to the order of those sentences or clauses.

The III-IV scores are lower, averaging 0.636. This reflects the difference between III's short, simple sentence style and IV's longer, more elaborated syntax. The result confirms that S distinguishes syntactic complexity from semantic similarity: III and IV can discuss the same topic while receiving lower structural similarity.

## Overall interpretation

The calibration produces the expected separation of dimensions:

- **O** detects whether ideas occur in a similar order. It strongly penalizes the reorganized IV sections.
- **L** detects shared vocabulary and phraseology. It is reduced by simplification and lexical substitution, but remains high when phrases are reused in a different order.
- **M** detects preservation of meaning. It remains high when content is reorganized, provided the propositions are retained.
- **S** detects similarity in syntactic complexity. It is reduced by the short, simple style of III but is relatively unaffected by sentence reordering in IV.

The most informative contrast is I-IV: low O combined with high L, M, and S. This demonstrates that OLMS is not collapsing all forms of similarity into one measure. It distinguishes a response that says similar things in a different organization from a response that changes vocabulary, simplifies syntax, or changes the amount of content.
