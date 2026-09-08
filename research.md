# OLMSVector Stability Pilot Study

This study treats an OLMSVector as

$$
\mathbf{v} = \langle O, L, M, S \rangle
$$

where $O$ is organization similarity, $L$ is lexical similarity, $M$ is meaning similarity, and $S$ is syntactic structure similarity between two language model responses.

The pilot goal is to estimate same-prompt response stability and prompt-induced response variation before defining a future equivalence threshold $\epsilon$. Because no threshold is defined yet, the analysis is primarily estimation-based rather than a formal equivalence test.

The study includes two prompts:

- Prompt A: the model is asked to provide rubric-based feedback.
- Prompt B: the model is asked to provide rubric-based feedback using simplified language for a lower-level language learner.

The same 10 essays are evaluated under both prompts. For each essay and each prompt, generate $K = 10$ model responses. The full data collection therefore contains:

$$
E \times P \times K = 10 \times 2 \times 10 = 200
$$

language model responses, where $E = 10$ essays and $P = 2$ prompts.

## Research Question 1

**When the prompt is unchanged and the essay is fixed, how much does the response vary?**

This is the within-essay stability question. The essay is held constant, the prompt is held constant, and the model is sampled repeatedly. This process should be performed separately for Prompt A and Prompt B. Any variation in the OLMSVectors reflects response-level instability under the same essay and prompt condition.

### Data Collection

Let there be $E = 10$ essays and two prompts $p \in \{A, B\}$. For each essay $e \in \{1, \ldots, E\}$ and prompt $p$, generate $K = 10$ responses.

For essay $e$ under prompt $p$, the responses are:

$$
r_{ep1}, r_{ep2}, \ldots, r_{epK}
$$

For each essay-prompt condition, compute all pairwise OLMSVectors among the $K$ responses:

$$
\mathbf{v}_{epij} = \langle O_{epij}, L_{epij}, M_{epij}, S_{epij} \rangle
\quad \text{for } 1 \leq i < j \leq K
$$

The number of pairwise comparisons per essay is:

$$
N = \binom{K}{2} = \frac{K(K - 1)}{2}
$$

With $K = 10$:

$$
N = \binom{10}{2} = 45
$$

For each prompt, across 10 essays, this produces:

$$
10 \times 45 = 450
$$

within-essay OLMSVectors. Across both prompts, this produces:

$$
2 \times 10 \times 45 = 900
$$

within-essay OLMSVectors.

### Analysis

For each essay $e$ and prompt $p$, compute the OLMS centroid:

$$
\mathbf{c}_{ep} =
\frac{1}{N}
\sum_{1 \leq i < j \leq K}
\mathbf{v}_{epij}
$$

Equivalently:

$$
\mathbf{c}_{ep} =
\frac{2}{K(K - 1)}
\sum_{1 \leq i < j \leq K}
\mathbf{v}_{epij}
$$

Then compute the distance from each pairwise vector to the essay-prompt-specific centroid:

$$
d_{epij} =
\left\|
\mathbf{v}_{epij} - \mathbf{c}_{ep}
\right\|
$$

Using Euclidean distance:

$$
d_{epij}
=
\sqrt{
(O_{epij} - c_{epO})^2
+ (L_{epij} - c_{epL})^2
+ (M_{epij} - c_{epM})^2
+ (S_{epij} - c_{epS})^2
}
$$

The mean within-essay distance is:

$$
\bar{d}_{ep} =
\frac{1}{N}
\sum_{1 \leq i < j \leq K}
d_{epij}
$$

$\bar{d}_{ep}$ is the primary within-essay stability estimate for essay $e$ under prompt $p$. Smaller values indicate that the repeated responses for the same essay and same prompt produce more tightly clustered OLMSVectors.

Also compute component-level summaries for each essay:

$$
\bar{O}_{ep} =
\frac{1}{N}
\sum_{1 \leq i < j \leq K}
O_{epij}
$$

$$
\bar{L}_{ep} =
\frac{1}{N}
\sum_{1 \leq i < j \leq K}
L_{epij}
$$

$$
\bar{M}_{ep} =
\frac{1}{N}
\sum_{1 \leq i < j \leq K}
M_{epij}
$$

$$
\bar{S}_{ep} =
\frac{1}{N}
\sum_{1 \leq i < j \leq K}
S_{epij}
$$

The dimensional analysis should also include variability estimates for each component, such as standard deviations and bootstrap confidence intervals. This is important because the total vector distance may appear stable even if one dimension, such as lexical similarity, varies more than the others.

### Bootstrap Confidence Intervals

Use a response-level bootstrap rather than treating the 45 pairwise comparisons as fully independent. The pairwise vectors are dependent because each model response appears in multiple pairwise comparisons.

For each essay $e$ and prompt $p$:

1. Sample $K = 10$ responses with replacement from the original set:

$$
\{r_{ep1}, r_{ep2}, \ldots, r_{epK}\}
$$

2. Recompute all pairwise OLMSVectors for the resampled responses.

3. Recompute the centroid $\mathbf{c}_{ep}^{*b}$, distances $d_{epij}^{*b}$, and mean distance $\bar{d}_{ep}^{*b}$.

4. Repeat for $B = 5000$ bootstrap replicates:

$$
b = 1, 2, \ldots, B
$$

The bootstrap distribution is:

$$
\bar{d}_{ep}^{*1}, \bar{d}_{ep}^{*2}, \ldots, \bar{d}_{ep}^{*B}
$$

A percentile-based 95% confidence interval is:

$$
\left[
Q_{0.025}\left(\bar{d}_{ep}^*\right),
Q_{0.975}\left(\bar{d}_{ep}^*\right)
\right]
$$

The same bootstrap approach can be used for each OLMS component:

$$
\bar{O}_{ep}^*, \bar{L}_{ep}^*, \bar{M}_{ep}^*, \bar{S}_{ep}^*
$$

This requires no additional language model calls. The language model is called only to generate the original $10 \times 2 \times 10 = 200$ responses. The 5000 bootstrap replicates are computational resamples of those responses and their derived OLMSVectors.

## Research Question 2

**Is the model equally stable across essays, or do different essays produce more variable feedback?**

This is the between-essay stability question. It should be answered separately for Prompt A and Prompt B. It does not compare responses from different essays directly. Instead, it compares the within-essay stability estimates across essays within each prompt condition.

The reason for this distinction is that different essays should often produce different feedback. Directly comparing a response to Essay A with a response to Essay B would confound essay content with response instability.

### Analysis

After answering Research Question 1, each essay has a within-essay stability estimate under each prompt. For a given prompt $p$, these estimates are:

$$
\bar{d}_{1p}, \bar{d}_{2p}, \ldots, \bar{d}_{Ep}
$$

where $E = 10$.

The overall mean within-essay instability across essays is:

$$
\bar{d}_{p,\text{overall}} =
\frac{1}{E}
\sum_{e=1}^{E}
\bar{d}_{ep}
$$

The between-essay variability in stability can be summarized as:

$$
s_{p,\text{between}} =
\sqrt{
\frac{1}{E - 1}
\sum_{e=1}^{E}
\left(
\bar{d}_{ep} - \bar{d}_{p,\text{overall}}
\right)^2
}
$$

Useful descriptive summaries include:

- minimum $\bar{d}_{ep}$
- maximum $\bar{d}_{ep}$
- median $\bar{d}_{ep}$
- interquartile range of $\bar{d}_{ep}$
- range of $\bar{d}_{ep}$
- coefficient of variation:

$$
CV =
\frac{s_{p,\text{between}}}{\bar{d}_{p,\text{overall}}}
$$

These summaries answer whether same-prompt stability is consistent across essays or whether some essays produce more variable feedback.

The same between-essay comparison should be performed for each OLMS component. For example, compare:

$$
\bar{O}_{1p}, \bar{O}_{2p}, \ldots, \bar{O}_{Ep}
$$

$$
\bar{L}_{1p}, \bar{L}_{2p}, \ldots, \bar{L}_{Ep}
$$

$$
\bar{M}_{1p}, \bar{M}_{2p}, \ldots, \bar{M}_{Ep}
$$

$$
\bar{S}_{1p}, \bar{S}_{2p}, \ldots, \bar{S}_{Ep}
$$

This can show whether instability differs by dimension. For example, feedback may be semantically stable but lexically variable.

### Interpretation

If the $\bar{d}_{ep}$ values are similar across essays for a given prompt, this suggests that same-prompt stability is relatively consistent across essay inputs under that prompt.

If some essays have much larger $\bar{d}_{ep}$ values, this suggests that those essays produce less stable feedback under that prompt. This may indicate that the essay is ambiguous, difficult to evaluate, near a rubric boundary, or otherwise more sensitive to model sampling variation.

Because this is a pilot study, these estimates can help define a future equivalence threshold $\epsilon$. A future confirmatory study could test:

$$
H_0: \bar{d} \geq \epsilon
$$

against:

$$
H_1: \bar{d} < \epsilon
$$

For the current pilot, however, the main outcome is an empirical estimate of the natural same-prompt OLMS variation.

## Research Question 4

**With the essay held constant, how much does changing from Prompt A to Prompt B change the model output, as measured by OLMSVectors?**

This is the between-prompt comparison question. It asks whether the simplified-language prompt changes the output beyond the ordinary response variation observed when each prompt is repeated under the same essay.

For each essay $e$, let the Prompt A responses be:

$$
A_e = \{a_{e1}, a_{e2}, \ldots, a_{eK}\}
$$

and let the Prompt B responses be:

$$
B_e = \{b_{e1}, b_{e2}, \ldots, b_{eK}\}
$$

The within-prompt Prompt A vectors are:

$$
\mathbf{v}^{AA}_{eij}
\quad \text{for } 1 \leq i < j \leq K
$$

The within-prompt Prompt B vectors are:

$$
\mathbf{v}^{BB}_{eij}
\quad \text{for } 1 \leq i < j \leq K
$$

The between-prompt vectors compare every Prompt A response to every Prompt B response for the same essay:

$$
\mathbf{v}^{AB}_{eij}
\quad \text{for } 1 \leq i \leq K,\ 1 \leq j \leq K
$$

For each essay, there are:

$$
\binom{K}{2} = 45
$$

within-prompt Prompt A comparisons,

$$
\binom{K}{2} = 45
$$

within-prompt Prompt B comparisons, and

$$
K^2 = 100
$$

between-prompt comparisons.

### Analysis

First compute the within-prompt centroids already defined in Research Question 1:

$$
\mathbf{c}_{eA}
$$

and

$$
\mathbf{c}_{eB}
$$

Then compute the same-prompt baseline centroid for essay $e$:

$$
\mathbf{c}_{e,\text{within}} =
\frac{\mathbf{c}_{eA} + \mathbf{c}_{eB}}{2}
$$

Next compute the between-prompt centroid for essay $e$:

$$
\mathbf{c}^{AB}_e =
\frac{1}{K^2}
\sum_{i=1}^{K}
\sum_{j=1}^{K}
\mathbf{v}^{AB}_{eij}
$$

The prompt-shift vector for essay $e$ is:

$$
\boldsymbol{\Delta}_e =
\mathbf{c}^{AB}_e - \mathbf{c}_{e,\text{within}}
$$

Because the OLMS components are similarity scores, negative values in $\boldsymbol{\Delta}_e$ indicate that Prompt A and Prompt B responses are less similar to each other than same-prompt responses are. The magnitude of the prompt-shift vector is:

$$
\delta_e =
\left\|
\boldsymbol{\Delta}_e
\right\|
$$

Using Euclidean distance:

$$
\delta_e =
\sqrt{
(\Delta_{eO})^2
+ (\Delta_{eL})^2
+ (\Delta_{eM})^2
+ (\Delta_{eS})^2
}
$$

Across essays, the mean prompt-induced shift is:

$$
\bar{\delta} =
\frac{1}{E}
\sum_{e=1}^{E}
\delta_e
$$

Interpretation:

- If $\delta_e \approx 0$, the Prompt A to Prompt B comparison is close to the same-prompt baseline for essay $e$.
- Larger values of $\delta_e$ indicate stronger prompt-induced output change for essay $e$.
- The signs of the components in $\boldsymbol{\Delta}_e$ show the direction of change. For similarity scores, negative component shifts mean that cross-prompt responses are less similar than same-prompt responses on that component.

### Bootstrap Confidence Intervals

Use a response-level bootstrap within each essay and prompt:

1. For each essay $e$, resample $K$ Prompt A responses with replacement from $A_e$.
2. Resample $K$ Prompt B responses with replacement from $B_e$.
3. Recompute within-prompt Prompt A vectors.
4. Recompute within-prompt Prompt B vectors.
5. Recompute $\mathbf{c}_{eA}^{*b}$, $\mathbf{c}_{eB}^{*b}$, and $\mathbf{c}_{e}^{AB,*b}$.
6. Compute:

$$
\mathbf{c}_{e,\text{within}}^{*b} =
\frac{\mathbf{c}_{eA}^{*b} + \mathbf{c}_{eB}^{*b}}{2}
$$

7. Compute:

$$
\boldsymbol{\Delta}_e^{*b} =
\mathbf{c}_{e}^{AB,*b} - \mathbf{c}_{e,\text{within}}^{*b}
$$

and:

$$
\delta_e^{*b} =
\left\|
\boldsymbol{\Delta}_e^{*b}
\right\|
$$

8. Average across essays:

$$
\bar{\delta}^{*b} =
\frac{1}{E}
\sum_{e=1}^{E}
\delta_e^{*b}
$$

9. Repeat for $B = 5000$ bootstrap replicates.

The bootstrap distribution is:

$$
\bar{\delta}^{*1}, \bar{\delta}^{*2}, \ldots, \bar{\delta}^{*B}
$$

A percentile-based 95% confidence interval is:

$$
\left[
Q_{0.025}\left(\bar{\delta}^*\right),
Q_{0.975}\left(\bar{\delta}^*\right)
\right]
$$

A directional bootstrap test can be reported for each component of $\boldsymbol{\Delta}$. For the scalar magnitude $\bar{\delta}$, the most useful pilot-study result is the effect estimate and confidence interval, because $\bar{\delta}$ is non-negative by construction.

For a component such as lexical similarity, a directional test could be:

$$
H_0: \bar{\Delta}_L \geq 0
$$

against:

$$
H_1: \bar{\Delta}_L < 0
$$

This asks whether cross-prompt lexical similarity is lower than the same-prompt lexical baseline. The bootstrap p-value-like quantity can be estimated as:

$$
p_{\text{boot}} =
\frac{1}{B}
\sum_{b=1}^{B}
I\left(\bar{\Delta}_L^{*b} \geq 0\right)
$$

where $I(\cdot)$ is an indicator function. The effect estimates and confidence intervals should remain the primary results, especially because this is a pilot study.

### Exploratory Multivariate Test

An exploratory one-sample Hotelling's $T^2$ test can also be used for RQ4. This tests whether the mean prompt-shift vector differs from the zero vector across essays.

The essay-level prompt-shift vectors are:

$$
\boldsymbol{\Delta}_1,
\boldsymbol{\Delta}_2,
\ldots,
\boldsymbol{\Delta}_E
$$

where each vector has $q = 4$ OLMS dimensions:

$$
\boldsymbol{\Delta}_e =
\langle
\Delta_{eO},
\Delta_{eL},
\Delta_{eM},
\Delta_{eS}
\rangle
$$

The null and alternative hypotheses are:

$$
H_0: \boldsymbol{\mu}_{\Delta} = \mathbf{0}
$$

against:

$$
H_1: \boldsymbol{\mu}_{\Delta} \neq \mathbf{0}
$$

The mean prompt-shift vector is:

$$
\bar{\boldsymbol{\Delta}} =
\frac{1}{E}
\sum_{e=1}^{E}
\boldsymbol{\Delta}_e
$$

Let $S_{\Delta}$ be the sample covariance matrix of the essay-level prompt-shift vectors. Hotelling's statistic is:

$$
T^2 =
E
\bar{\boldsymbol{\Delta}}^\top
S_{\Delta}^{-1}
\bar{\boldsymbol{\Delta}}
$$

This can be converted to an $F$ statistic:

$$
F =
\frac{E - q}{q(E - 1)}
T^2
$$

with degrees of freedom:

$$
q,\ E - q
$$

For this study, $E = 10$ and $q = 4$, so:

$$
F =
\frac{10 - 4}{4(10 - 1)}
T^2
=
\frac{1}{6}T^2
$$

with degrees of freedom:

$$
4,\ 6
$$

This test should be treated as exploratory rather than primary. The design produces only $E = 10$ essay-level prompt-shift vectors, and Hotelling's $T^2$ requires estimating and inverting a $4 \times 4$ covariance matrix. With such a small number of essays, the covariance estimate may be unstable. Therefore, the primary RQ4 evidence should remain the effect estimates, component-level shifts, and bootstrap confidence intervals.

### Dimensional Analysis

Repeat the RQ4 contrast separately for each OLMS component. For essay $e$:

$$
\Delta_{eO} =
\bar{O}^{AB}_e -
\frac{\bar{O}^{AA}_e + \bar{O}^{BB}_e}{2}
$$

$$
\Delta_{eL} =
\bar{L}^{AB}_e -
\frac{\bar{L}^{AA}_e + \bar{L}^{BB}_e}{2}
$$

$$
\Delta_{eM} =
\bar{M}^{AB}_e -
\frac{\bar{M}^{AA}_e + \bar{M}^{BB}_e}{2}
$$

$$
\Delta_{eS} =
\bar{S}^{AB}_e -
\frac{\bar{S}^{AA}_e + \bar{S}^{BB}_e}{2}
$$

The essay-level component shifts can then be averaged across essays:

$$
\bar{\Delta}_O =
\frac{1}{E}
\sum_{e=1}^{E}
\Delta_{eO}
$$

and similarly for $\bar{\Delta}_L$, $\bar{\Delta}_M$, and $\bar{\Delta}_S$.

For the simplified-language prompt, a substantively meaningful pattern may be larger negative shifts in lexical similarity and syntactic structure similarity, but smaller shifts in meaning and organization. That would suggest that Prompt B changes wording and sentence structure while preserving the core feedback content.

## Follow-Up Question 3

**Does the model produce appropriately different feedback for meaningfully different essays?**

This is a discriminability or sensitivity question, not a stability question.

To answer it, a future analysis would need to compare within-essay similarity against between-essay similarity.

Within-essay comparisons measure repeated responses to the same essay:

$$
\mathbf{v}_{eij}
\quad \text{for responses } i,j \text{ to the same essay } e
$$

Between-essay comparisons would compare responses generated for different essays:

$$
\mathbf{v}_{e f i j}
\quad \text{for } e \neq f
$$

where response $i$ comes from essay $e$, and response $j$ comes from essay $f$.

If the OLMSVector is behaving usefully, then responses to the same essay should generally be more similar to each other than responses to different essays. In distance terms, one would expect:

$$
d_{\text{within}} < d_{\text{between}}
$$

or, if using similarity scores directly:

$$
\text{similarity}_{\text{within}} >
\text{similarity}_{\text{between}}
$$

This analysis would require constructing cross-essay response comparisons and comparing their distribution to the within-essay comparison distribution.

For example:

$$
\Delta =
\bar{d}_{\text{between}} - \bar{d}_{\text{within}}
$$

where larger positive values of $\Delta$ would indicate stronger discrimination between same-essay and different-essay feedback.

This question should be handled separately from the stability pilot because different essays are expected to produce different feedback. Cross-essay differences should not be interpreted as prompt instability; they are evidence about whether the model and OLMSVector can distinguish meaningfully different essay inputs.
