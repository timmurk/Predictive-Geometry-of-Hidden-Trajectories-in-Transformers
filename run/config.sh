export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}"
OUT=${OUT:-outputs}
DEVICE=${DEVICE:-cuda}

MODELS=(
  "HuggingFaceTB/SmolLM2-1.7B"
  "meta-llama/Llama-3.2-1B"
  "Qwen/Qwen2.5-3B"
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-v0.1"
  "meta-llama/Llama-2-7b-hf"
  "google/gemma-2-2b"
  "google/gemma-7b"
  "allenai/OLMo-3-7B"
  "google/gemma-2-9b"
)
DATASETS=(wikitext openwebtext fineweb)
CHECK_MODELS=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1")
KD_MODELS=("Qwen/Qwen2.5-0.5B" "HuggingFaceTB/SmolLM2-135M" "HuggingFaceTB/SmolLM2-360M")

short() { basename "$1"; }

if [ -n "${ONLY_MODEL:-}" ]; then MODELS=("$ONLY_MODEL"); fi
if [ -n "${ONLY_DATASET:-}" ]; then DATASETS=("$ONLY_DATASET"); fi
