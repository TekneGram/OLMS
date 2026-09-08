O_old: Used hard alignment to perform pair matching.
However, the BERTScores for the kind of feedback in student essays are too noisy.
This leads to poor matching later.
So, here we use BERTScore's full matrix to softly determine how parts are organized.

Soft Alignment

  Assume two texts are split into ordered units:

  A = [A1, A2, A3]
  B = [B1, B2, B3]

  Each unit may be a sentence, clause, paragraph chunk, or other segment.

  The goal is to measure whether units in A tend to correspond to the same relative
  locations in B, without forcing brittle one-to-one matches.

  Step 1: Build A BERTScore Matrix

  Compute pairwise BERTScore similarity between every unit in A and every unit in B.

  Example:

          B1    B2    B3
  A1    0.90  0.35  0.20
  A2    0.40  0.75  0.55
  A3    0.10  0.45  0.85

  Call this matrix:

  S_ij = similarity(A_i, B_j)

  Step 2: Convert Each Row Into A Softmax Distribution

  For each unit A_i, convert its similarities to all B_j into weights:

  W_ij = exp(S_ij / T) / sum_k exp(S_ik / T)

  where T is the softmax temperature.

  - Lower T: sharper, closer to hard matching.
  - Higher T: flatter, more tolerant of uncertainty.

  Example with moderate temperature:

          B1    B2    B3
  A1    0.67  0.21  0.12
  A2    0.20  0.41  0.39
  A3    0.10  0.25  0.65

  Each row sums to 1.

  Interpretation:

  A1 softly aligns mostly to B1.
  A2 is split between B2 and B3.
  A3 softly aligns mostly to B3.

  Step 3: Compute Expected Position In B

  Assign normalized positions to units.

  pos_A = [0.00, 0.50, 1.00]
  pos_B = [0.00, 0.50, 1.00]

  For each A_i, compute the expected position in B:

  E_B(i) = sum_j W_ij * pos_B(j)

  Example:

  E_B(A1) = 0.67*0.00 + 0.21*0.50 + 0.12*1.00 = 0.225
  E_B(A2) = 0.20*0.00 + 0.41*0.50 + 0.39*1.00 = 0.595
  E_B(A3) = 0.10*0.00 + 0.25*0.50 + 0.65*1.00 = 0.775

  So:

  E_B = [0.225, 0.595, 0.775]

  Step 4: Calculate Directional Position Similarity

  Compare each unit’s position in A to where it softly lands in B:

  p_i = 1 - |pos_A(i) - E_B(i)|

  Example:

  A1: 1 - |0.00 - 0.225| = 0.775
  A2: 1 - |0.50 - 0.595| = 0.905
  A3: 1 - |1.00 - 0.775| = 0.775

  Unweighted directional score:

  A_to_B = mean([0.775, 0.905, 0.775]) = 0.818

  Step 5: Weight By Confidence

  A softmax distribution can be misleading if all similarities are weak or ambiguous. For
  example:

  [0.34, 0.33, 0.33]

  This does not mean the unit strongly maps to the middle; it means the model is uncertain.

  So weight each unit by confidence. A useful confidence measure is based on entropy.

  Entropy measures how spread out a distribution is:

  H_i = - sum_j W_ij * log(W_ij)

  Maximum entropy occurs when the distribution is perfectly flat:

  H_max = log(n_B)

  Normalize and invert it:

  conf_i = 1 - H_i / log(n_B)

  Interpretation:

  conf_i near 1 = sharp, confident alignment
  conf_i near 0 = flat, uncertain alignment

  Example:

  A1 weights = [0.67, 0.21, 0.12]
  H(A1) ≈ 0.85
  conf(A1) = 1 - 0.85/log(3) ≈ 0.23

  A2 weights = [0.20, 0.41, 0.39]
  H(A2) ≈ 1.06
  conf(A2) = 1 - 1.06/log(3) ≈ 0.04

  A3 weights = [0.10, 0.25, 0.65]
  H(A3) ≈ 0.86
  conf(A3) = 1 - 0.86/log(3) ≈ 0.22

  Then compute the confidence-weighted directional score:

  A_to_B =
    sum_i conf_i * p_i
    / sum_i conf_i

  Example:

  A_to_B =
    (0.23*0.775 + 0.04*0.905 + 0.22*0.775)
    / (0.23 + 0.04 + 0.22)

  A_to_B ≈ 0.786

  Here A2 contributes less because its distribution is ambiguous.

  Step 6: Make The Score Symmetric

  Everything above maps A into B. But similarity should not depend on which text is first.

  So repeat the process in the reverse direction:

  B_to_A

  Use the transposed similarity matrix, softmax over each B_j’s similarities to all A_i,
  compute expected positions in A, calculate position similarity, and weight by confidence.

  Example:

  A_to_B = 0.786
  B_to_A = 0.812

  Final soft organization score:

  SoftOrg(A, B) = (A_to_B + B_to_A) / 2

  Example:

  SoftOrg(A, B) = (0.786 + 0.812) / 2 = 0.799

  Overall Formula

  Given similarity matrix S, temperature T, unit positions x_i for A, and y_j for B:

  W_ij = exp(S_ij / T) / sum_k exp(S_ik / T)

  E_B(i) = sum_j W_ij * y_j

  p_i = 1 - |x_i - E_B(i)|

  H_i = - sum_j W_ij log(W_ij)

  conf_i = 1 - H_i / log(n_B)

  A_to_B =
    sum_i conf_i p_i
    / sum_i conf_i

  SoftOrg(A, B) =
    1/2 * (A_to_B + B_to_A)

  This produces a score in [0, 1], where higher means the same content tends to appear in
  the same relative regions of both texts, without requiring exact hard unit matches.