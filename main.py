from deprel import DependencyParser

def main() -> None:
  my_text = "The thesis statement clearly presents your opinion, but it needs to explain why the topic matters. Although your thesis statement is focused, it does not fully preview the reasons you discuss in the essay. Your thesis says that school uniforms are good, but it shoul state your exact position more clearly. To make the thesis stronger, add a reason that connects your opinion to the main argument. Your thesis is understandable because it gives the reader a clear opinion. The thesis would be clearer if you named the two reasons that your body paragraph will explain. You have a thesis statement, but because it is very general, the reader may not know your main argument. Your thesis statement, which appears at the end of the introduction, gives an opinion but does not include a clear reason. I suggest revising the thesis so that it includes both your position and the reason for that position. The thesis is not specific enough to guide the rest of the essay."
  deprel = DependencyParser(
    ".models/udpipe"
  )

  parsed = deprel.parse("trial", my_text)
  print(parsed)

  return

if __name__ == "__main__":
  main()
