from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RUNTIME_DIR = Path(
    "/mnt/ssd/Agent/D-Robotics_LLM_S100_1.0.0_SDK/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime"
)
DEFAULT_MODEL_API_URL = "http://127.0.0.1:18081/infer"
DEFAULT_MODEL_NAME = "oellm_multichat_worker"


@dataclass
class AgentConfig:
    runtime_dir: Path
    multichat_bin: Path
    multichat_cfg: Path
    run_bin: Path
    hbm_path: Path
    tokenizer_dir: Path
    template_path: Path
    model_type: int
    model_api_url: str
    model_name: str
    api_key: str
    enable_local_model: bool
    local_model_workers: int
    workdir: Path
    kb_db: Path
    kb_collection: str
    kb_embed_model: str
    kb_rerank_model: str
    http_host: str
    http_port: int
    local_model_base_port: int
    llm_only: bool
    llm_skip_when_risk_normal: bool
    max_requests_per_session: int
    max_session_age_sec: float


def build_config(args: argparse.Namespace) -> AgentConfig:
    runtime_dir = Path(args.runtime_dir).resolve()
    multichat_bin = (
        Path(args.multichat_bin).resolve()
        if args.multichat_bin
        else runtime_dir / "example/oellm_multichat/build/oellm_multichat"
    )
    multichat_cfg = (
        Path(args.multichat_cfg).resolve()
        if args.multichat_cfg
        else runtime_dir / "example/oellm_multichat/qwen_multichat_config_3b.json"
    )
    run_bin = (
        Path(args.run_bin).resolve()
        if args.run_bin
        else runtime_dir / "example/oellm_agent_run/build/oellm_agent_run"
    )
    hbm_path = (
        Path(args.hbm_path).resolve()
        if args.hbm_path
        else runtime_dir / "model/Qwen2.5_1.5B_Instruct_1024.hbm"
    )
    tokenizer_dir = (
        Path(args.tokenizer_dir).resolve()
        if args.tokenizer_dir
        else runtime_dir / "config/Qwen2.5_1.5B_Instruct_config"
    )
    template_path = (
        Path(args.template_path).resolve()
        if args.template_path
        else runtime_dir / "config/Qwen2.5_1.5B_Instruct_config/Qwen2.5_1.5B_Instruct.jinja"
    )
    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()
    kb_db = Path(args.kb_db).resolve() if args.kb_db else (workdir / "kb/api/kb/data/kb.sqlite3").resolve()
    return AgentConfig(
        runtime_dir=runtime_dir,
        multichat_bin=multichat_bin,
        multichat_cfg=multichat_cfg,
        run_bin=run_bin,
        hbm_path=hbm_path,
        tokenizer_dir=tokenizer_dir,
        template_path=template_path,
        model_type=args.model_type,
        model_api_url=args.model_api_url,
        model_name=args.model_name,
        api_key=args.api_key,
        enable_local_model=args.enable_local_model,
        local_model_workers=max(1, int(args.local_model_workers)),
        workdir=workdir,
        kb_db=kb_db,
        kb_collection=args.kb_collection,
        kb_embed_model=args.kb_embed_model,
        kb_rerank_model=args.kb_rerank_model,
        http_host="0.0.0.0",
        http_port=18080,
        local_model_base_port=int(getattr(args, "local_model_base_port", 18081)),
        llm_only=bool(getattr(args, "llm_only", False)),
        llm_skip_when_risk_normal=bool(getattr(args, "llm_skip_when_risk_normal", True)),
        max_requests_per_session=max(1, int(getattr(args, "max_requests_per_session", 24) or 24)),
        max_session_age_sec=max(60.0, float(getattr(args, "max_session_age_sec", 300.0) or 300.0)),
    )
