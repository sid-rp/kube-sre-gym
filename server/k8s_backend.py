"""
Kubernetes API backend — auth, command dispatch, cluster reset, and health checks.

Command execution is delegated to k8s_commands.CommandHandler.
Failure injection is delegated to k8s_injectors.FailureInjector.
"""

import os
import logging
import subprocess
import time

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .constants import APP_NAMESPACES, HEALTHY_STATE, RESET_POLL_INTERVAL, RESET_MAX_POLLS
from .k8s_commands import CommandHandler, _pod_status
from .k8s_injectors import FailureInjector

logger = logging.getLogger(__name__)


def _load_token_auth(endpoint: str, ca_cert_b64: str, token: str):
    """Authenticate to K8s with a bearer token + CA cert.

    Works anywhere — Docker, HF Spaces, Colab, H100. No GCP SDK needed.
    """
    import tempfile
    import base64
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    ca_cert = base64.b64decode(ca_cert_b64)
    ca_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    ca_path.write(ca_cert)
    ca_path.close()

    configuration = client.Configuration()
    configuration.host = endpoint
    configuration.ssl_ca_cert = ca_path.name
    configuration.api_key = {"BearerToken": token}
    configuration.api_key_prefix = {"BearerToken": "Bearer"}
    client.Configuration.set_default(configuration)


class K8sBackend:
    """Gateway to a live Kubernetes cluster.

    Auth:  Token-based (GKE) or in-cluster (when running inside K8s).
    Commands:  Delegated to CommandHandler (k8s_commands.py).
    Injection: Delegated to FailureInjector (k8s_injectors.py).
    Reset:     Patches deployments back to HEALTHY_STATE.
    """

    def __init__(self):
        self.manifests_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_app")
        self.app_namespaces = APP_NAMESPACES

        # Auth priority: kubeconfig > in-cluster > token-based
        # kubeconfig has full RBAC (dev/Jupyter), token may have limited permissions
        try:
            config.load_kube_config()
            logger.info("K8s auth: kubeconfig")
        except config.ConfigException:
            try:
                config.load_incluster_config()
                logger.info("K8s auth: in-cluster")
            except config.ConfigException:
                endpoint = os.environ.get("K8S_ENDPOINT")
                ca_cert_b64 = os.environ.get("K8S_CA_CERT")
                token = os.environ.get("K8S_TOKEN")
                if endpoint and ca_cert_b64 and token:
                    _load_token_auth(endpoint, ca_cert_b64, token)
                    logger.info(f"K8s auth: token-based ({endpoint})")
                else:
                    raise RuntimeError(
                        "No K8s auth available. Set kubeconfig, run in-cluster, "
                        "or set K8S_ENDPOINT + K8S_TOKEN + K8S_CA_CERT env vars."
                    )

        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

        self.commands = CommandHandler(self.v1, self.apps_v1)
        self.injector = FailureInjector(self.v1, self.apps_v1, self.commands)

    def execute(self, command: str) -> str:
        """Parse and execute a kubectl-style command on the real cluster."""
        cmd = command.strip()
        if cmd.startswith("kubectl "):
            cmd = cmd[8:]

        parts = cmd.split()
        if not parts:
            return "error: empty command"

        verb = parts[0]
        ns = self._parse_namespace(parts)

        # Strip namespace flags from parts before passing to handler
        cleaned = []
        skip = False
        for p in parts[1:]:
            if skip:
                skip = False
                continue
            if p == "-n":
                skip = True
                continue
            if p in ("--all-namespaces", "-A"):
                continue
            cleaned.append(p)

        try:
            return self.commands.dispatch(verb, cleaned, ns, raw_cmd=cmd)
        except ApiException as e:
            return f"Error from server ({e.reason}): {e.body}"
        except Exception as e:
            logger.error(f"Execute error: {e}", exc_info=True)
            return f"ERROR: {str(e)}"

    def inject_failure(self, failure_type: str, params: dict) -> str:
        """Inject a real failure into the GKE cluster."""
        return self.injector.inject(failure_type, params)

    def reset(self):
        """Reset cluster to healthy state.

        Uses HEALTHY_STATE to patch deployments back to known-good config via K8s API.
        For hackathon namespace (not in HEALTHY_STATE), falls back to kubectl apply.
        Also clears injected resource quotas and bad liveness probes.
        """
        # Step 1: Reset core namespaces (payments, frontend, auth) via HEALTHY_STATE
        for ns, deployments in HEALTHY_STATE.items():
            try:
                self.v1.read_namespace(ns)
            except ApiException:
                logger.warning(f"Namespace '{ns}' not found — skipping reset for it")
                continue

            for deploy_name, spec in deployments.items():
                try:
                    deploy = self.apps_v1.read_namespaced_deployment(deploy_name, ns)
                except ApiException:
                    logger.warning(f"Deployment '{deploy_name}' not found in '{ns}' — skipping")
                    continue

                try:
                    probe_spec = spec.get("liveness_probe")
                    for c in deploy.spec.template.spec.containers:
                        c.image = spec["image"]
                        if not c.resources:
                            c.resources = client.V1ResourceRequirements()
                        c.resources.limits = {"memory": spec["memory_limit"]}
                        c.env = [client.V1EnvVar(name=k, value=v)
                                 for k, v in spec["env"].items()]
                        c.command = None
                        c.args = None
                        # Restore liveness probe from HEALTHY_STATE
                        if probe_spec:
                            c.liveness_probe = client.V1Probe(
                                http_get=client.V1HTTPGetAction(
                                    path=probe_spec["path"],
                                    port=probe_spec["port"],
                                ),
                                initial_delay_seconds=10,
                                period_seconds=10,
                            )
                        elif c.liveness_probe:
                            # Deployment shouldn't have a probe — remove any injected one
                            c.liveness_probe = None

                    deploy.spec.replicas = spec["replicas"]
                    self.apps_v1.replace_namespaced_deployment(deploy_name, ns, deploy)
                    logger.info(f"Reset {deploy_name} in {ns} to healthy state")
                except ApiException as e:
                    logger.error(f"Failed to reset {deploy_name} in {ns}: {e.reason}")

            # Clean up injected resource quotas
            try:
                quotas = self.v1.list_namespaced_resource_quota(ns)
                for q in quotas.items:
                    self.v1.delete_namespaced_resource_quota(q.metadata.name, ns)
                    logger.info(f"Removed quota '{q.metadata.name}' from {ns}")
            except ApiException:
                pass

        # Step 2: Reset hackathon namespace via manifests (if present)
        base_dir = os.path.join(self.manifests_dir, "base", "hackathon")
        if os.path.exists(base_dir):
            ns_file = os.path.join(self.manifests_dir, "namespaces.yaml")
            if os.path.exists(ns_file):
                subprocess.run(["kubectl", "apply", "-f", ns_file],
                               capture_output=True, timeout=30)
            subprocess.run(["kubectl", "apply", "-R", "-f", base_dir],
                           capture_output=True, timeout=60)

        # Wait for pods to stabilize
        for _ in range(RESET_MAX_POLLS):
            health = self.check_health()
            all_healthy = all(
                s in ("Running", "Completed")
                for ns_pods in health.values() for s in ns_pods.values()
            ) if health else False
            if all_healthy and health:
                logger.info("Cluster reset complete — all pods healthy")
                return
            time.sleep(RESET_POLL_INTERVAL)
        logger.warning("Cluster reset timed out — some pods may not be healthy")

    def check_health(self) -> dict:
        """Return {namespace: {pod_name: status}} for all app namespaces."""
        health = {}
        for ns in self.app_namespaces:
            try:
                pods = self.v1.list_namespaced_pod(ns)
                health[ns] = {
                    p.metadata.name: _pod_status(p)
                    for p in pods.items
                }
            except ApiException as e:
                logger.error(f"check_health: failed to list pods in '{ns}': {e.reason}")
                health[ns] = {}
        return health

    @staticmethod
    def _parse_namespace(parts: list[str]) -> str | None:
        for i, p in enumerate(parts):
            if p == "-n" and i + 1 < len(parts):
                return parts[i + 1]
        if "--all-namespaces" in parts or "-A" in parts:
            return "__all__"
        return None
