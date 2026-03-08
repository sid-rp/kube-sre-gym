"""
FastAPI application for the K8s SRE Environment.

Usage:
    uvicorn k8s_sre_env.server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app
from ..models import K8sSREAction, K8sSREObservation
from .k8s_environment import K8sEnvironment

app = create_app(K8sEnvironment, K8sSREAction, K8sSREObservation, env_name="k8s-sre-env")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
