"""
Failure injectors — injects real faults into a GKE cluster for SRE training.

Each injector modifies a real K8s deployment to simulate a production incident.
The CommandHandler's set_resources/set_image/set_env methods are reused where possible.

Container names are looked up from HEALTHY_STATE to match actual deployment specs.
"""

import logging
import time

from kubernetes import client
from kubernetes.client.rest import ApiException

from .constants import INJECT_WAIT_DEFAULT, HEALTHY_STATE

logger = logging.getLogger(__name__)


def _get_container_name(ns: str, deploy: str) -> str:
    """Look up the container name from HEALTHY_STATE, falling back to deploy name."""
    ns_state = HEALTHY_STATE.get(ns, {})
    deploy_state = ns_state.get(deploy, {})
    return deploy_state.get("container_name", deploy)


class FailureInjector:
    """Injects failures into a running K8s cluster."""

    def __init__(self, v1: client.CoreV1Api, apps_v1: client.AppsV1Api, cmd_handler):
        self.v1 = v1
        self.apps_v1 = apps_v1
        self.cmd = cmd_handler  # CommandHandler instance for set_resources/set_env/etc.

    def inject(self, failure_type: str, params: dict) -> str:
        """Inject a failure by type. Returns a status message."""
        ns = params.get("namespace", "payments")
        deploy = params.get("deployment", "payment-api")

        injectors = {
            "oom_kill": self._inject_oom,
            "crashloop": self._inject_crashloop,
            "image_pull": self._inject_image_pull,
            "bad_config": self._inject_bad_config,
            "resource_quota": self._inject_resource_quota,
            "scale_zero": self._inject_scale_zero,
        }
        fn = injectors.get(failure_type)
        if not fn:
            return f"Unknown failure type: {failure_type}"
        return fn(ns, deploy)

    def _inject_oom(self, ns: str, deploy: str) -> str:
        container_name = _get_container_name(ns, deploy)
        self.cmd.set_resources(
            [f"deployment/{deploy}", "-c", container_name, "--limits=memory=4Mi"], ns)
        time.sleep(INJECT_WAIT_DEFAULT)
        return f"Injected OOMKill on {deploy} in {ns}"

    def _inject_crashloop(self, ns: str, deploy: str) -> str:
        try:
            d = self.apps_v1.read_namespaced_deployment(deploy, ns)
            d.spec.template.spec.containers[0].command = ["sh", "-c", "exit 1"]
            self.apps_v1.replace_namespaced_deployment(deploy, ns, d)
            time.sleep(10)
            return f"Injected CrashLoopBackOff on {deploy} in {ns}"
        except ApiException as e:
            return f"Error injecting crashloop: {e.reason}"

    def _inject_image_pull(self, ns: str, deploy: str) -> str:
        try:
            d = self.apps_v1.read_namespaced_deployment(deploy, ns)
            d.spec.template.spec.containers[0].image = "nginx:nonexistent-tag-99999"
            self.apps_v1.replace_namespaced_deployment(deploy, ns, d)
            time.sleep(5)
            return f"Injected ImagePullBackOff on {deploy} in {ns}"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _inject_bad_config(self, ns: str, deploy: str) -> str:
        # payment-worker checks DATABASE_URL, not DB_HOST
        if deploy == "payment-worker":
            self.cmd.set_env([f"deployment/{deploy}",
                              "DATABASE_URL=postgres://wrong-host.invalid:5432/payments"], ns)
        else:
            self.cmd.set_env([f"deployment/{deploy}", "DB_HOST=wrong-host.invalid.local"], ns)
        time.sleep(INJECT_WAIT_DEFAULT)
        return f"Injected bad config on {deploy} in {ns}"

    def _inject_resource_quota(self, ns: str, _deploy: str) -> str:
        quota = client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name="tight-quota", namespace=ns),
            spec=client.V1ResourceQuotaSpec(hard={"pods": "1", "requests.memory": "32Mi"}),
        )
        try:
            self.v1.create_namespaced_resource_quota(ns, quota)
            return f"Injected tight ResourceQuota in {ns}"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _inject_scale_zero(self, ns: str, deploy: str) -> str:
        try:
            d = self.apps_v1.read_namespaced_deployment(deploy, ns)
            d.spec.replicas = 0
            self.apps_v1.replace_namespaced_deployment(deploy, ns, d)
            time.sleep(5)
            return f"Injected scale-to-zero on {deploy} in {ns}"
        except ApiException as e:
            return f"Error: {e.reason}"

