# Configuration file for running experiments

# ---------------------
# LLM Server Config
# ---------------------
llm_path = "/Users/danielmikaleola/Documents/Development/EssayLensPython/.appdata/models/Qwen3-4B-Q8_0.gguf"
n_ctx=4096
n_threads=8
n_gpu_layers=999
verbose=False

# ---------------------
# LLM Response Config
# ---------------------
max_tokens=512
temperature=0.0
top_p=0.95
stream=False

# ---------------------
# Clause Pair-Matcher
# ---------------------
capacity=2
min_score=0.2

# ---------------------
# Essay Feedback Config
# ---------------------
essays_path = ".essays/"
# System Prompt
ai_role = "You are a tutor for English as a foreign language students. "
ai_task_main = "You will read an essay and provide evaluation of the essay's thesis statement. "
ai_knowledge = (
  "The thesis statement is part of the introduction. It often comes towards the end of the introduction. A good thesis statement has some of the following properties: \n",
  "Main idea: A thesis statement should state the main idea of the essay, and this is essential.\n",
  "Opinion: A thesis statement can state the opinion of the writer, though it is not essential to do this.\n",
  "Preview: A thesis statement can preview the content of the essay, though it is not essential to do this.\n",
  "Length: A thesis statement should not be too complex.\n"
  )
ai_task_description = "Provide feedback on the properties of the writer's thesis statement in the essay provided by the student: \n\n"

ai_recipient_information_1 = "Be thorough in your feedback and provide advice if necessary. The students are CEFR level B1, so use simple sentence and vocabulary in your feedback."
ai_recipient_information_2 = "Be thorough in your feedback and provide advice if necessary. The students are CEFR level C1."
