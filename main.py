from deprel import DependencyParser

def main() -> None:
  my_text = "This is a lovely text. I would like you to parse the dependency relations in this text so that I can see whether or not it is working. Go for it!"
  deprel = DependencyParser(
    ".models/udpipe"
  )

  parsed = deprel.parse("trial", my_text)
  print(parsed)

  return

if __name__ == "__main__":
  main()
