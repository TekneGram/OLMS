# Configuration file for running experiments

# ---------------------
# LLM Server Config
# ---------------------
llm_path = "/Users/danielmikaleola/Documents/Development/EssayLensPython/.appdata/models/Qwen3-4B-Q8_0.gguf"
n_ctx=4096
n_threads=8
n_gpu_layers=999
verbose=False
seed=42

# ---------------------
# LLM Response Config
# ---------------------
max_tokens=512
temperature=0.7
top_p=0.95
stream=True

# ---------------------
# Clause Pair-Matcher
# ---------------------
capacity=1
min_score=0.4

# ---------------------
# Essay Feedback Config
# ---------------------
essays_path = ".essays/"
# System Prompt
ai_role = "You are a tutor for English as a foreign language students. "
#ai_task_main = "You will read an essay and provide evaluation of the essay's thesis statement. Provide JSON format, no markdown: {'main_idea': '...', 'opinion': '...', 'preview': '...', 'length': '...', 'advice': '...'}"
ai_task_main = "You will read an essay and provide evaluation of the essay's thesis statement. Provide plain text feedback in full sentences with sentence punctuation. No markdown, no bullet points, no colons."
ai_knowledge = (
  "The thesis statement is part of the introduction. It often comes towards the end of the introduction. A good thesis statement has some of the following properties: \n",
  "Main idea: A thesis statement should state the main idea of the essay with a specific focus, and this is essential.\n",
  "Opinion: A thesis statement can state the opinion of the writer, though it is not essential to do this.\n",
  "Preview: A thesis statement can preview the content of the essay by introducing topics developed in the body paragraphs, though it is not essential to do this.\n",
  "Length: A thesis statement should not be too complex.\n"
  )
ai_task_description = "Evaluate the properties of the writer's thesis statement in the essay provided by the student: \n\n"

ai_recipient_information_1 = "Write detailed feedback for each property of the thesis statement. Provide advice."
ai_recipient_information_2 = "Write one sentence of feedback per property of the thesis statement. Provide advice in one sentence."

ai_response_format = {
  "type": "json_object",
  "json_object": {
    "name": "feedback",
    "schema": {
      "type": "object",
      "properties": {
        "main_idea": { "type" : "string" },
        "opinion": { "type": "string" },
        "preview": { "type": "string"},
        "length": { "type" : "string" },
        "advice": { "type": "string" }
      },
      "required": ["main_idea", "opinion", "preview", "length", "advice"],
      "additionalProperties": False
    }
  }
}

ai_response_required_fields = [
  "main_idea",
  "opinion",
  "preview",
  "length",
  "advice"
]