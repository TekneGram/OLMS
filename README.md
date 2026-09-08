# OLMS
Evaluate prompt outputs on small language models for variation.
O = Organization similarity
L = Lexical similarity
M = Meaning / Semantic similarity
S = Structural similarity

## Workflow

Run commands from the repository root with the project environment activated:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The three stages run independently. No stage automatically starts another.
Running `python main.py` displays help without loading any models.
The implemented analyses follow RQ1, RQ2, and RQ4 in [research.md](research.md).

### 1. Collect Responses

Set the model path, generation settings, and the two prompt instructions in
`configuration.py`. A uses `ai_recipient_information_1`; B uses
`ai_recipient_information_2`. The program uses these texts as configured.

```bash
python main.py responses --repeats 10 --output data/pilot_responses.csv
```

Every `.md` essay in the configured essay directory receives 10 responses per
prompt by default. Essay count is discovered dynamically: E essays produce
`E * 2 * 10` responses. Use `--essays PATH` to select a different directory.
At least two repetitions per prompt are required for this study.

Each response is saved immediately. A companion `pilot_responses.metadata.json`
records prompt text, model settings, essay content hashes, repetition count, and
collection completion. Each response gets a reproducible seed derived from the
configured base seed, essay filename, prompt label, and repetition index.
Seeds preserve sampling identity on resumption; exact model output reproducibility
also depends on the model, runtime, and hardware.

```bash
python main.py responses --repeats 10 --output data/pilot_responses.csv --resume
```

Resume skips completed responses and can collect newly added essays. Existing
essays, prompt text, repetition count, and generation settings must still match.
Use a new output filename for a changed experiment. Original model files must
also be kept consistent; model binaries are not copied into experiment metadata.

Response CSV columns: `essay_file`, `prompt` (`A` or `B`), `response_index`
(1-based within each essay and prompt), and `response`.

### 2. Calculate OLMS Vectors

After response collection finishes:

```bash
python main.py vectors --responses data/pilot_responses.csv --output data/pilot_vectors.csv
```

This stage reads saved responses and loads UDPipe and BERTScore, without making
generation-model calls. Responses are parsed once per essay during the run.
For K responses per prompt, each essay has:

| Comparison | Study pairs | At K = 10 |
|---|---:|---:|
| AA | K(K - 1)/2 | 45 |
| BB | K(K - 1)/2 | 45 |
| AB | K squared | 100 |
| Self comparisons, bootstrap support only | 2K | 20 |

The CSV therefore contains 210 rows per essay at K = 10: 190 study vectors and
20 self-comparison vectors. Self scores are calculated using the actual OLMS
measures, because comparing a response to itself need not yield four ones.
They are excluded from the original study summaries.

Vector columns: `essay_file`, `prompt_1`, `response_index_1`, `prompt_2`,
`response_index_2`, `comparison_type`, `response_count`, `org`, `lex`, `meaning`,
and `struct`. AA/BB indices are ordered with the first no greater than the second;
equal indices identify bootstrap support rows. AB always places A first.

Pairs are saved immediately. Add `--resume` to the same command after an
interruption. Resuming requires unchanged response input and vector code/settings.
After adding responses or essays, generate a new vector output file.
Use `--parser-dir PATH` to select the UDPipe model directory, and `--diagnostics`
to append sentence and structure Markdown logs alongside the vectors. Diagnostic
logs may repeat entries after resumption; the vector CSV does not duplicate pairs.

### 3. Analyze Saved Vectors

```bash
python main.py analyze --vectors data/pilot_vectors.csv --output data/pilot_analysis --bootstrap-replicates 5000 --seed 42
```

Analysis loads numerical libraries only. It requires complete AA, BB, AB, and
self-comparison coverage for every essay. It uses raw OLMS components, Euclidean
distances, sample SDs (`ddof=1`), and percentile bootstrap intervals (95% by
default; configurable with `--confidence`). A different essay count is supported
without changes to code or formulas.

Bootstrap samples draw response IDs with replacement within each essay and prompt.
Saved scores reconstruct all sample pairs, preserving repeated selections and
self comparisons. A and B are resampled separately; essays remain fixed. The
overall prompt-shift magnitude is the average of essay magnitudes, not the
magnitude of the average shift vector.

The output directory contains:

| Output | Contents |
|---|---|
| `within_essay_stability.csv` | RQ1 centroids, component SDs, mean centroid distances, and counts |
| `vector_distances.csv` | Each original AA/BB vector's distance to its centroid |
| `within_essay_intervals.csv` | RQ1 bootstrap intervals for mean distances and component means |
| `between_essay_stability.csv` | RQ2 mean, SD, extrema, quartiles, median, IQR, range, and CV across essays |
| `prompt_shift.csv` | RQ4 essay-level centroids, baselines, signed component shifts, and magnitudes |
| `overall_prompt_shift.csv` | Mean component shifts and mean magnitude across essays |
| `prompt_shift_intervals.csv`, `overall_prompt_shift_intervals.csv` | RQ4 intervals; overall component directional bootstrap quantities |
| `*_bootstrap.csv.gz` | Individual bootstrap estimates, with replicate IDs |
| `hotelling_t2.json` | Exploratory T-squared/F test or the reason it is unavailable |
| `metadata.json`, `report.md` | Analysis settings, input fingerprint/provenance, and readable results |

Undefined descriptive statistics are empty CSV cells (for example, CV when the
mean is zero, or sample SD with one essay). Undefined JSON statistics are null.
Hotelling's test requires more than four essays and a usable covariance matrix;
singular or numerically ill-conditioned cases are reported as unavailable.
Its independence and multivariate normality assumptions warrant caution in a
small pilot. Directional bootstrap quantities are the protocol's fractions of
component replicates at or above zero, not calibrated confirmatory p-values.
Effect estimates and confidence intervals remain primary. The future equivalence
threshold and cross-essay discriminability question are outside this analysis.

Existing response/vector files are protected against accidental overwrite.
Analysis requires a new or empty output directory. Legacy response CSVs without
prompt labels and legacy vector CSVs without response pair IDs cannot support this
study and produce an actionable error. The examples above use new filenames so
existing `feedback_data` outputs can remain in place.

### Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover numerical calculations, response-level bootstrap reconstruction,
pair counts, incremental collection, interruption/resumption, malformed inputs,
and an analysis CLI run with 5,000 bootstrap replicates. Generation and vector
workflow tests use mocked model/scorer calls and do not require model downloads.

## Dependency Parsing

`deprel.py` provides a small UDPipe-based dependency parsing step for the pipeline.

`DependencyParser` accepts raw text and returns a `DependencyParse` object:

```python
from deprel import DependencyParser

parser = DependencyParser()
parsed = parser.parse("trial", "This is a lovely text.")

print(parsed.name)
print(parsed.conllu)
print(parsed.tokens)
```

`DependencyParse` stores the original text, the raw CoNLL-U output, and a list of `DependencyToken` objects. Each token includes its sentence-local `token_id`, generated `global_token_id`, POS tags, lemma, head token, and dependency relation.

The UDPipe model downloads on first use into `.models/udpipe/`. This folder is ignored by Git, so model files are available locally without being uploaded to GitHub.
