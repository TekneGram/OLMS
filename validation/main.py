from create_data_set import CreateDataSet

def create_data_set() -> None:
  data = CreateDataSet()
  data.scan_for_files(corpus_path="/Volumes/Corpora/BAWE_TXT")
  data.sample_corpus_texts(seed=42)
  data.extract_text(seed=42)


def main() -> None:
  
  return

if __name__ == "__main__":
  main()

