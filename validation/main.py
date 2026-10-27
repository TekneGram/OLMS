from create_data_set import CreateDataSet
from parse_data_set import ParseDataSet

def create_data_set() -> None:
  data = CreateDataSet()
  data.scan_for_files(corpus_path="/Volumes/Corpora/BAWE_TXT")
  data.sample_corpus_texts(seed=42)
  data.extract_text(seed=42)

def parse_data_set() -> None:
  pds = ParseDataSet()
  pds.parse()


def main() -> None:
  # Step 1
  # Call create_data_set on a corpus of BAWE text txt files
  # Step 2
  # Call parse_data_set on the files created in step 1 to create partner texts where sentences are either the same or re-ordered in some way.
  
  return

if __name__ == "__main__":
  main()

