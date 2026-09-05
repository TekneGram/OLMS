# OLMS
Evaluate prompt outputs on small language models for variation.
O = Organization similarity
L = Lexical similarity
M = Meaning / Semantic similarity
S = Structural similarity

## Workflow
SLM Output --> Dependency relation tagging --> clause splitting -->
BERTScore clauses --> match pairs --> O - L - M - S calculations --->
Statistical analysis

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
