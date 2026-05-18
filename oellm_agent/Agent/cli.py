from __future__ import annotations

import argparse
from .config import DEFAULT_MODEL_API_URL, DEFAULT_MODEL_NAME, DEFAULT_RUNTIME_DIR, build_config
from .server import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Board-side Python Agent based on oellm_multichat")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR), help="oellm_runtime path")
    parser.add_argument("--multichat-bin", default="", help="path to oellm_multichat binary")
    parser.add_argument("--multichat-cfg", default="", help="path to multichat config json")
    parser.add_argument("--run-bin", default="", help="path to oellm_run binary")
    parser.add_argument("--hbm-path", default="", help="path to model hbm file")
    parser.add_argument("--tokenizer-dir", default="", help="path to tokenizer dir")
    parser.add_argument("--template-path", default="", help="path to chat template jinja")
    parser.add_argument("--model-type", type=int, default=7, help="model type for oellm_run")
    parser.add_argument("--workdir", default="", help="agent allowed workspace root")
    parser.add_argument("--kb-db", default="", help="sqlite metadata db path (default: <workdir>/kb/data/kb.sqlite3)")
    parser.add_argument("--kb-collection", default="coal_truck_kb", help="faiss index name")
    parser.add_argument("--kb-embed-model", default="/mnt/ssd/Agent/modelscope/hub/models/BAAI/bge-small-zh-v1___5", help="sentence-transformers embedding model local directory")
    parser.add_argument("--kb-rerank-model", default="/mnt/ssd/Agent/modelscope/hub/models/BAAI/bge-reranker-base", help="sentence-transformers reranker model local directory")
    parser.add_argument("--model-api-url", default=DEFAULT_MODEL_API_URL, help="direct model API URL, e.g. http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="model name for API payload")
    parser.add_argument("--api-key", default="", help="optional API key")
    parser.add_argument("--enable-local-model", action="store_true", help="use local oellm_multichat process")
    parser.add_argument("--local-model-workers", type=int, default=3, help="number of local model workers (default: 3)")
    parser.add_argument("--local-model-base-port", type=int, default=18081, help="base port for spawned local HTTP model workers")
    parser.add_argument("--llm-only", action="store_true", help="disable rule fallback/guardrails and show raw LLM decisions")
    parser.add_argument("--model-timeout-sec", type=float, default=5.0, help="model request timeout (seconds)")
    args = parser.parse_args()
    cfg = build_config(args)
    if not args.enable_local_model and not cfg.model_api_url:
        raise ValueError("either --enable-local-model or --model-api-url is required")
    run_agent(cfg)

if __name__ == "__main__":
    main()