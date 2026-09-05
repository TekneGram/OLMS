from dataclasses import dataclass
from typing import List
from pathlib import Path
import os
import urllib.request

from ufal.udpipe import Model, Pipeline, ProcessingError

DEFAULT_MODEL_URL = (
  "https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/"
  "master/inst/udpipe-ud-2.5-191206/english-ewt-ud-2.5-191206.udpipe"
)

DEFAULT_MODEL_NAME = "english-ewt-ud-2.5-191206.udpipe"

@dataclass
class DependencyToken:
  sentence_id: int
  token_id: int
  global_token_id: int
  text: str
  lemma: str
  upos: str
  xpos: str
  head_token_id: int | None
  dependency_relation: str

@dataclass
class DependencyParse:
  name: str
  source_text: str
  conllu: str
  tokens: List[DependencyToken]

class DependencyParser:
  def __init__(
      self, 
      model_dir: str | Path | None = None,
      model_url: str = DEFAULT_MODEL_URL,
      model_name: str = DEFAULT_MODEL_NAME,
      ) -> None:
    
    self.model_dir = Path(
      model_dir
      or os.environ.get("UDPIPE_MODEL_DIR", ".models/udpipe")
    )
    self.model_url = model_url
    self.model_path = self.model_dir / model_name

    self.model_dir.mkdir(parents=True, exist_ok=True)
    self._download_model_if_missing()

    self.model = Model.load(str(self.model_path))
    if self.model is None:
      raise RuntimeError(f"Could not load UDPipe model: {self.model_path}")

    self.pipeline = Pipeline(
      self.model,
      "tokenize",
      Pipeline.DEFAULT,
      Pipeline.DEFAULT,
      "conllu"
    )

    self.parsed: DependencyParse | None = None

  def _download_model_if_missing(self) -> None:
    if self.model_path.exists():
      return

    print(f"Downloading UDPipe model to {self.model_path}")
    urllib.request.urlretrieve(self.model_url, self.model_path)

  def parse(self, name: str, text: str) -> DependencyParse:
    error = ProcessingError()
    conllu = self.pipeline.process(text, error)

    if error.occurred():
      raise RuntimeError(error.message)

    self.parsed = DependencyParse(
      name=name,
      source_text=text,
      conllu=conllu,
      tokens=self._tokens_from_conllu(conllu)
    )
    return self.parsed

  def get_parsed(self) -> DependencyParse | None:
    return self.parsed

  def _tokens_from_conllu(self, conllu: str) -> list[DependencyToken]:
    tokens = []
    current_sentence_id = 0
    global_token_id = 0

    for line in conllu.strip().splitlines():
      line = line.strip()

      if not line:
        continue

      if line.startswith("# sent_id ="):
        current_sentence_id = int(line.split("=", 1)[1].strip())
        continue

      if line.startswith("#"):
        continue

      columns = line.split("\t")
      if len(columns) != 10:
        continue

      token_id, text, lemma, upos, xpos, _, head, deprel, _, _ = columns

      # Skip inferred words and multiword token range
      if "-" in token_id or "." in token_id:
        continue

      global_token_id += 1

      tokens.append(
        DependencyToken(
          sentence_id = current_sentence_id,
          token_id=int(token_id),
          global_token_id=global_token_id,
          text=text,
          lemma=lemma,
          upos=upos,
          xpos=xpos,
          head_token_id=None if head == "0" else int(head),
          dependency_relation=deprel,
        )
      )

    return tokens
