# Soft Organization Mathematics

Assume text \(A\) has \(m\) ordered units and text \(B\) has \(n\) ordered units:

$$
A = [A_1, A_2, \dots, A_m]
$$

$$
B = [B_1, B_2, \dots, B_n]
$$

The BERTScore table gives a similarity score between every unit in \(A\) and every unit in \(B\). Using one selected metric, usually F1, define the similarity matrix:

$$
S \in \mathbb{R}^{m \times n}
$$

where:

$$
S_{ij} = \operatorname{sim}(A_i, B_j)
$$

## Normalized Positions

Each unit receives a normalized position in its own text.

For text \(A\):

$$
x_i =
\begin{cases}
0, & m = 1 \\
\dfrac{i}{m - 1}, & m > 1
\end{cases}
$$

For text \(B\):

$$
y_j =
\begin{cases}
0, & n = 1 \\
\dfrac{j}{n - 1}, & n > 1
\end{cases}
$$

Here \(i\) and \(j\) are zero-based indices, so positions range from \(0\) to \(1\).

## Softmax Alignment

For each unit \(A_i\), convert its similarities to all units in \(B\) into a probability distribution:

$$
W_{ij}
=
\frac{\exp(S_{ij} / T)}
{\sum_{k=1}^{n} \exp(S_{ik} / T)}
$$

where \(T > 0\) is the softmax temperature.

Lower \(T\) makes the distribution sharper:

$$
T \downarrow \quad \Rightarrow \quad W_i \text{ approaches hard matching}
$$

Higher \(T\) makes the distribution flatter:

$$
T \uparrow \quad \Rightarrow \quad W_i \text{ becomes more tolerant and uncertain}
$$

Each row sums to \(1\):

$$
\sum_{j=1}^{n} W_{ij} = 1
$$

## Expected Position

The expected position in \(B\) for unit \(A_i\) is the weighted average of all positions in \(B\):

$$
E_B(i) = \sum_{j=1}^{n} W_{ij} y_j
$$

This says where \(A_i\) softly lands in text \(B\).

## Directional Position Score

Compare the original position of \(A_i\) with its expected position in \(B\):

$$
p_i = 1 - |x_i - E_B(i)|
$$

Since \(x_i\) and \(E_B(i)\) are both in \([0, 1]\), the position score is also in \([0, 1]\):

$$
0 \le p_i \le 1
$$

A value near \(1\) means the unit lands in a similar relative location. A value near \(0\) means it lands far away.

## Entropy Confidence

A softmax distribution can be sharp or flat. Sharp distributions indicate stronger alignment confidence. Flat distributions indicate ambiguity.

The entropy of the alignment distribution for \(A_i\) is:

$$
H_i = -\sum_{j=1}^{n} W_{ij} \log(W_{ij})
$$

The maximum entropy occurs when the distribution is perfectly flat:

$$
H_{\max} = \log(n)
$$

The confidence score is normalized by maximum entropy and inverted:

$$
c_i = 1 - \frac{H_i}{\log(n)}
$$

So:

$$
c_i \approx 1
\quad \text{means confident, sharp alignment}
$$

$$
c_i \approx 0
\quad \text{means uncertain, flat alignment}
$$

When \(n = 1\), there is only one possible target unit, so the implementation treats confidence as:

$$
c_i = 1
$$

## Directional Organization Score

The directional organization score from \(A\) to \(B\) is the confidence-weighted average of the position scores:

$$
O_{A \to B}
=
\frac{\sum_{i=1}^{m} c_i p_i}
{\sum_{i=1}^{m} c_i}
$$

If all confidence values are zero, the implementation falls back to the unweighted mean:

$$
O_{A \to B}
=
\frac{1}{m}\sum_{i=1}^{m} p_i
$$

## Symmetric Organization Score

The same calculation is repeated in the reverse direction by using the transposed similarity matrix:

$$
S^\top \in \mathbb{R}^{n \times m}
$$

This gives:

$$
O_{B \to A}
$$

The final soft organization score is the average of the two directional scores:

$$
\operatorname{SoftOrg}(A, B)
=
\frac{1}{2}
\left(
O_{A \to B}
+
O_{B \to A}
\right)
$$

The final score is clamped to the interval \([0, 1]\):

$$
0 \le \operatorname{SoftOrg}(A, B) \le 1
$$

Higher values mean that similar content tends to appear in similar relative regions of both texts, without requiring brittle one-to-one pair matches.
