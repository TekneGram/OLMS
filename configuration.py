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