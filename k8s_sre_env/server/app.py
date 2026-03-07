from openenv.core.env_server import create_app
from .k8s_environment import K8sEnvironment
from ..models import K8sSREAction, K8sSREObservation

app = create_app(K8sEnvironment(), K8sSREAction, K8sSREObservation)
