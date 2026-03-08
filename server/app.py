"""
FastAPI application for the Kube SRE Gym Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from ..models import KubeSreGymAction, KubeSreGymObservation
    from .kube_sre_gym_environment import KubeSreGymEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import KubeSreGymAction, KubeSreGymObservation
    from server.kube_sre_gym_environment import KubeSreGymEnvironment

import logging

logger = logging.getLogger(__name__)

app = create_app(
    KubeSreGymEnvironment,
    KubeSreGymAction,
    KubeSreGymObservation,
    env_name="kube_sre_gym",
    max_concurrent_envs=1,
)


@app.get("/healthz")
async def healthz():
    """Quick health check — tests if environment can be created."""
    try:
        env = KubeSreGymEnvironment()
        health = env.backend.check_health()
        return {"status": "ok", "cluster_health": health}
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for `uv run server` and direct execution."""
    import argparse
    import os
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Kube SRE Gym server")
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--host", default=host)
    parser.add_argument("--gym-mode", choices=("standard", "adversarial"), default=None,
                        help="Override GYM_MODE env var")
    parser.add_argument("--llm-backend", choices=("openai", "hf", "anthropic"), default=None,
                        help="Override LLM_BACKEND env var")
    parser.add_argument("--llm-model", default=None,
                        help="Override LLM_MODEL env var (e.g. claude-sonnet-4-20250514)")
    parser.add_argument("--anthropic-api-key", default=None,
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    if args.gym_mode:
        os.environ["GYM_MODE"] = args.gym_mode
    if args.llm_backend:
        os.environ["LLM_BACKEND"] = args.llm_backend
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
