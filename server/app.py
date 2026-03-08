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

app = create_app(
    KubeSreGymEnvironment,
    KubeSreGymAction,
    KubeSreGymObservation,
    env_name="kube_sre_gym",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
