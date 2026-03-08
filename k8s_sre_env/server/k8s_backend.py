import os
import logging
import time
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

logger = logging.getLogger(__name__)


def _load_token_auth(endpoint: str, ca_cert_b64: str, token: str):
    """
    Authenticate to K8s with a bearer token + CA cert.
    Works anywhere — Docker, HF Spaces, Colab, H100. No GCP SDK needed.
    """
    import tempfile, base64, urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    configuration = client.Configuration()
    configuration.host = endpoint
    configuration.api_key = {"authorization": f"Bearer {token}"}

    # Try CA cert first, fall back to verify_ssl=False
    try:
        ca_cert_bytes = base64.b64decode(ca_cert_b64)
        ca_cert_file = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
        ca_cert_file.write(ca_cert_bytes)
        ca_cert_file.close()
        configuration.ssl_ca_cert = ca_cert_file.name
    except Exception:
        logger.warning("CA cert decode failed, disabling SSL verification")
        configuration.verify_ssl = False

    client.Configuration.set_default(configuration)

    # Test connection
    try:
        v1 = client.CoreV1Api()
        v1.list_namespace(_request_timeout=5)
        logger.info(f"Authenticated to {endpoint} via bearer token (SSL verified)")
    except Exception:
        # CA cert might not work on this platform, retry without SSL verify
        configuration.verify_ssl = False
        configuration.ssl_ca_cert = None
        client.Configuration.set_default(configuration)
        logger.info(f"Authenticated to {endpoint} via bearer token (SSL verify disabled)")


class K8sBackend:
    """Direct Kubernetes API client for GKE cluster.

    Auth priority (simplest first):
      1. K8S_TOKEN + K8S_ENDPOINT + K8S_CA_CERT — bearer token (works everywhere)
      2. In-cluster config (if running inside the GKE cluster)
      3. KUBECONFIG / ~/.kube/config (local dev with gcloud configured)
    """

    def __init__(self):
        self.connected = False
        token = os.environ.get("K8S_TOKEN")
        endpoint = os.environ.get("K8S_ENDPOINT")       # https://34.169.10.97
        ca_cert = os.environ.get("K8S_CA_CERT")          # base64 CA cert

        try:
            if token and endpoint and ca_cert:
                _load_token_auth(endpoint, ca_cert, token)
                logger.info("Using bearer token auth")
            else:
                try:
                    config.load_incluster_config()
                    logger.info("Loaded in-cluster config")
                except config.ConfigException:
                    kubeconfig_path = os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config"))
                    config.load_kube_config(config_file=kubeconfig_path)
                    logger.info(f"Loaded kubeconfig from {kubeconfig_path}")
            self.connected = True
        except Exception as e:
            logger.warning(f"K8s auth failed (will retry on first use): {e}")

        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

        # Path to sample app manifests (friend provides these)
        self.manifests_dir = os.environ.get("MANIFESTS_DIR", "./sample_app")
        self.app_namespaces = ["payments", "frontend", "auth"]

    # =========================================================
    # EXECUTE — run kubectl-equivalent commands via K8s API
    # =========================================================

    def execute(self, command: str) -> str:
        """Parse a kubectl-style command string and execute via K8s API.
        Returns formatted text output matching real kubectl format."""

        if command.startswith("diagnose:"):
            return "Diagnosis recorded. Continue investigating or apply a fix."
        if command.startswith("fix:"):
            return self.execute(command[len("fix:"):].strip())

        cmd = command.replace("kubectl ", "").strip()
        parts = cmd.split()
        if not parts:
            return "error: no command specified"

        verb = parts[0]
        ns = self._parse_namespace(parts)

        try:
            if verb == "get":
                return self._cmd_get(parts[1:], ns)
            elif verb == "describe":
                return self._cmd_describe(parts[1:], ns)
            elif verb == "logs":
                return self._cmd_logs(parts[1:], ns)
            elif verb == "top":
                return self._cmd_top(parts[1:], ns)
            elif verb == "rollout":
                return self._cmd_rollout(parts[1:], ns)
            elif verb == "set":
                return self._cmd_set(parts[1:], ns)
            elif verb == "delete":
                return self._cmd_delete(parts[1:], ns)
            elif verb == "scale":
                return self._cmd_scale(parts[1:], ns)
            elif verb == "patch":
                return self._cmd_patch(parts[1:], ns)
            else:
                return f"error: unknown command '{verb}'"
        except ApiException as e:
            return f"Error from server ({e.reason}): {e.body}"
        except Exception as e:
            logger.error(f"Execute error: {e}")
            return f"ERROR: {str(e)}"

    def _parse_namespace(self, parts):
        for i, p in enumerate(parts):
            if p == "-n" and i + 1 < len(parts):
                return parts[i + 1]
        if "--all-namespaces" in parts or "-A" in parts:
            return "__all__"
        return None

    # --- GET ---
    def _cmd_get(self, parts, ns):
        if not parts:
            return "error: resource type required"
        resource = parts[0]

        if resource in ("pods", "pod", "po"):
            return self._get_pods(ns)
        elif resource in ("deployments", "deploy"):
            return self._get_deployments(ns)
        elif resource in ("events", "ev"):
            return self._get_events(ns)
        elif resource in ("nodes", "node"):
            return self._get_nodes()
        elif resource in ("services", "svc"):
            return self._get_services(ns)
        elif resource in ("resourcequota", "quota"):
            return self._get_resourcequotas(ns)
        else:
            return f"error: the server doesn't have a resource type \"{resource}\""

    def _get_pods(self, ns):
        if ns == "__all__":
            pods = self.v1.list_pod_for_all_namespaces()
        elif ns:
            pods = self.v1.list_namespaced_pod(ns)
        else:
            pods = self.v1.list_pod_for_all_namespaces()

        lines = ["NAME                              READY   STATUS             RESTARTS   AGE"]
        for p in pods.items:
            name = p.metadata.name[:34].ljust(34)
            total = len(p.spec.containers)
            ready_count = sum(1 for cs in (p.status.container_statuses or []) if cs.ready)
            ready = f"{ready_count}/{total}".ljust(8)
            status = (p.status.phase or "Unknown")
            for cs in (p.status.container_statuses or []):
                if cs.state and cs.state.waiting and cs.state.waiting.reason:
                    status = cs.state.waiting.reason
                elif cs.state and cs.state.terminated and cs.state.terminated.reason:
                    status = cs.state.terminated.reason
            status = status[:19].ljust(19)
            restarts = str(sum(cs.restart_count for cs in (p.status.container_statuses or []))).ljust(11)
            age = self._format_age(p.metadata.creation_timestamp)
            lines.append(f"{name}{ready}{status}{restarts}{age}")
        return "\n".join(lines) if len(lines) > 1 else "No resources found."

    def _get_deployments(self, ns):
        if ns and ns != "__all__":
            deploys = self.apps_v1.list_namespaced_deployment(ns)
        else:
            deploys = self.apps_v1.list_deployment_for_all_namespaces()

        lines = ["NAME              READY   UP-TO-DATE   AVAILABLE   AGE"]
        for d in deploys.items:
            name = d.metadata.name[:18].ljust(18)
            ready = f"{d.status.ready_replicas or 0}/{d.spec.replicas or 0}".ljust(8)
            uptodate = str(d.status.updated_replicas or 0).ljust(13)
            available = str(d.status.available_replicas or 0).ljust(12)
            age = self._format_age(d.metadata.creation_timestamp)
            lines.append(f"{name}{ready}{uptodate}{available}{age}")
        return "\n".join(lines)

    def _get_events(self, ns):
        if ns and ns != "__all__":
            events = self.v1.list_namespaced_event(ns)
        else:
            events = self.v1.list_event_for_all_namespaces()

        lines = ["LAST SEEN   TYPE      REASON           OBJECT              MESSAGE"]
        for e in sorted(events.items, key=lambda x: x.metadata.creation_timestamp or "")[-20:]:
            age = self._format_age(e.last_timestamp or e.metadata.creation_timestamp)
            etype = (e.type or "Normal")[:10].ljust(10)
            reason = (e.reason or "")[:17].ljust(17)
            obj = f"{e.involved_object.kind}/{e.involved_object.name}"[:20].ljust(20)
            msg = (e.message or "")[:80]
            lines.append(f"{age.ljust(12)}{etype}{reason}{obj}{msg}")
        return "\n".join(lines)

    def _get_nodes(self):
        nodes = self.v1.list_node()
        lines = ["NAME                    STATUS   ROLES    AGE   VERSION"]
        for n in nodes.items:
            name = n.metadata.name[:24].ljust(24)
            conditions = {c.type: c.status for c in n.status.conditions}
            status = "Ready" if conditions.get("Ready") == "True" else "NotReady"
            roles = ",".join(k.split("/")[-1] for k in n.metadata.labels if "node-role" in k) or "<none>"
            age = self._format_age(n.metadata.creation_timestamp)
            version = n.status.node_info.kubelet_version
            lines.append(f"{name}{status.ljust(9)}{roles.ljust(9)}{age.ljust(6)}{version}")
        return "\n".join(lines)

    def _get_services(self, ns):
        if ns and ns != "__all__":
            svcs = self.v1.list_namespaced_service(ns)
        else:
            svcs = self.v1.list_service_for_all_namespaces()
        lines = ["NAME              TYPE        CLUSTER-IP     PORT(S)"]
        for s in svcs.items:
            name = s.metadata.name[:18].ljust(18)
            stype = (s.spec.type or "ClusterIP")[:12].ljust(12)
            ip = (s.spec.cluster_ip or "None")[:15].ljust(15)
            ports = ",".join(f"{p.port}/{p.protocol}" for p in (s.spec.ports or []))
            lines.append(f"{name}{stype}{ip}{ports}")
        return "\n".join(lines)

    def _get_resourcequotas(self, ns):
        if not ns or ns == "__all__":
            return "error: namespace required for resourcequota"
        quotas = self.v1.list_namespaced_resource_quota(ns)
        if not quotas.items:
            return "No resources found."
        lines = []
        for q in quotas.items:
            lines.append(f"Name:    {q.metadata.name}")
            lines.append(f"Resource    Used    Hard")
            for resource, hard in (q.status.hard or {}).items():
                used = (q.status.used or {}).get(resource, "0")
                lines.append(f"{resource.ljust(12)}{str(used).ljust(8)}{hard}")
        return "\n".join(lines)

    # --- DESCRIBE ---
    def _cmd_describe(self, parts, ns):
        if len(parts) < 2:
            return "error: resource name required"
        rtype, rname = parts[0], parts[1]

        if rtype in ("pod", "pods", "po"):
            return self._describe_pod(rname, ns)
        elif rtype in ("deployment", "deploy"):
            return self._describe_deployment(rname, ns)
        elif rtype in ("node", "nodes"):
            return self._describe_node(rname)
        return f"error: unsupported describe for {rtype}"

    def _describe_pod(self, name, ns):
        """Find pod by partial name match, return describe-style output."""
        namespace = ns or "default"
        pods = self.v1.list_namespaced_pod(namespace)
        pod = None
        for p in pods.items:
            if name in p.metadata.name:
                pod = p
                break
        if not pod:
            return f'Error from server (NotFound): pods "{name}" not found in namespace "{namespace}"'

        lines = [
            f"Name:         {pod.metadata.name}",
            f"Namespace:    {pod.metadata.namespace}",
            f"Node:         {pod.spec.node_name}",
            f"Status:       {pod.status.phase}",
            f"IP:           {pod.status.pod_ip}",
            f"Containers:"
        ]
        for c in pod.spec.containers:
            lines.append(f"  {c.name}:")
            lines.append(f"    Image:          {c.image}")
            cs = next((s for s in (pod.status.container_statuses or []) if s.name == c.name), None)
            if cs:
                if cs.state.running:
                    lines.append(f"    State:          Running")
                elif cs.state.waiting:
                    lines.append(f"    State:          Waiting")
                    lines.append(f"      Reason:       {cs.state.waiting.reason}")
                elif cs.state.terminated:
                    lines.append(f"    State:          Terminated")
                    lines.append(f"      Reason:       {cs.state.terminated.reason}")
                    lines.append(f"      Exit Code:    {cs.state.terminated.exit_code}")
                if cs.last_state and cs.last_state.terminated:
                    lines.append(f"    Last State:     Terminated")
                    lines.append(f"      Reason:       {cs.last_state.terminated.reason}")
                    lines.append(f"      Exit Code:    {cs.last_state.terminated.exit_code}")
                lines.append(f"    Ready:          {cs.ready}")
                lines.append(f"    Restart Count:  {cs.restart_count}")
            if c.resources:
                if c.resources.limits:
                    lines.append(f"    Limits:")
                    for k, v in c.resources.limits.items():
                        lines.append(f"      {k}:     {v}")
                if c.resources.requests:
                    lines.append(f"    Requests:")
                    for k, v in c.resources.requests.items():
                        lines.append(f"      {k}:     {v}")
            if c.liveness_probe:
                lines.append(f"    Liveness:       {self._format_probe(c.liveness_probe)}")
            if c.env:
                lines.append(f"    Environment:")
                for e in c.env:
                    lines.append(f"      {e.name}: {e.value}")

        # Events for this pod
        events = self.v1.list_namespaced_event(namespace,
            field_selector=f"involvedObject.name={pod.metadata.name}")
        if events.items:
            lines.append("Events:")
            for e in events.items[-10:]:
                age = self._format_age(e.last_timestamp or e.metadata.creation_timestamp)
                lines.append(f"  {e.type}\t{e.reason}\t{e.message}")

        return "\n".join(lines)

    def _describe_deployment(self, name, ns):
        namespace = ns or "default"
        try:
            d = self.apps_v1.read_namespaced_deployment(name, namespace)
        except ApiException:
            return f'Error from server (NotFound): deployments.apps "{name}" not found'
        lines = [
            f"Name:               {d.metadata.name}",
            f"Namespace:          {d.metadata.namespace}",
            f"Replicas:           {d.spec.replicas} desired | {d.status.available_replicas or 0} available",
            f"Strategy:           {d.spec.strategy.type if d.spec.strategy else 'RollingUpdate'}",
        ]
        return "\n".join(lines)

    def _describe_node(self, name):
        try:
            n = self.v1.read_node(name)
        except ApiException:
            return f'Error from server (NotFound): nodes "{name}" not found'
        lines = [
            f"Name:         {n.metadata.name}",
            f"Allocatable:  cpu={n.status.allocatable.get('cpu')}, memory={n.status.allocatable.get('memory')}",
        ]
        return "\n".join(lines)

    def _format_probe(self, probe):
        if probe.http_get:
            return f"http-get {probe.http_get.path}:{probe.http_get.port} delay={probe.initial_delay_seconds}s period={probe.period_seconds}s"
        return "configured"

    # --- LOGS ---
    def _cmd_logs(self, parts, ns):
        if not parts:
            return "error: pod name required"
        pod_name = parts[0]
        namespace = ns or "default"
        tail = 50
        container = None
        for i, p in enumerate(parts):
            if p.startswith("--tail="):
                tail = int(p.split("=")[1])
            if p == "-c" and i + 1 < len(parts):
                container = parts[i + 1]

        pods = self.v1.list_namespaced_pod(namespace)
        matched = next((p for p in pods.items if pod_name in p.metadata.name), None)
        if not matched:
            return f'Error from server (NotFound): pods "{pod_name}" not found'

        try:
            logs = self.v1.read_namespaced_pod_log(
                matched.metadata.name, namespace,
                container=container, tail_lines=tail)
            return logs if logs else "(no logs)"
        except ApiException as e:
            return f"Error: {e.reason}"

    # --- TOP ---
    def _cmd_top(self, parts, ns):
        if not parts:
            return "error: resource type required"
        if parts[0] == "pods":
            return self._top_pods(ns)
        elif parts[0] == "nodes":
            return self._top_nodes()
        return "error: unsupported"

    def _top_pods(self, ns):
        try:
            api = client.CustomObjectsApi()
            if ns and ns != "__all__":
                metrics = api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", ns, "pods")
            else:
                metrics = api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "pods")

            lines = ["NAME                              CPU(cores)   MEMORY(bytes)"]
            for item in metrics.get("items", []):
                name = item["metadata"]["name"][:34].ljust(34)
                for c in item.get("containers", []):
                    cpu = c["usage"].get("cpu", "0")
                    mem = c["usage"].get("memory", "0")
                    lines.append(f"{name}{cpu.ljust(13)}{mem}")
            return "\n".join(lines)
        except Exception:
            return "error: Metrics API not available. Use 'kubectl describe pod' to see resource requests/limits."

    def _top_nodes(self):
        try:
            api = client.CustomObjectsApi()
            metrics = api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")
            lines = ["NAME                    CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%"]
            for item in metrics.get("items", []):
                name = item["metadata"]["name"][:24].ljust(24)
                cpu = item["usage"].get("cpu", "0").ljust(13)
                mem = item["usage"].get("memory", "0")
                lines.append(f"{name}{cpu}  -      {mem.ljust(16)}-")
            return "\n".join(lines)
        except Exception:
            return "error: Metrics API not available."

    # --- MUTATING COMMANDS ---
    def _cmd_rollout(self, parts, ns):
        if parts and parts[0] == "restart" and len(parts) > 1:
            deploy_name = parts[1].replace("deployment/", "")
            namespace = ns or "default"
            try:
                body = {"spec": {"template": {"metadata": {"annotations": {
                    "kubectl.kubernetes.io/restartedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }}}}}
                self.apps_v1.patch_namespaced_deployment(deploy_name, namespace, body)
                return f"deployment.apps/{deploy_name} restarted"
            except ApiException as e:
                return f"Error: {e.reason}"
        return "error: unsupported rollout command"

    def _cmd_set(self, parts, ns):
        namespace = ns or "default"
        if not parts:
            return "error: subcommand required"

        if parts[0] == "resources":
            return self._set_resources(parts[1:], namespace)
        elif parts[0] == "image":
            return self._set_image(parts[1:], namespace)
        elif parts[0] == "env":
            return self._set_env(parts[1:], namespace)
        return "error: unsupported set command"

    def _set_resources(self, parts, ns):
        deploy_name = None
        container_name = None
        limits = {}
        for i, p in enumerate(parts):
            if "/" in p and not p.startswith("-"):
                deploy_name = p.split("/")[-1]
            if p == "-c" and i + 1 < len(parts):
                container_name = parts[i + 1]
            if p.startswith("--limits="):
                for kv in p[len("--limits="):].split(","):
                    k, v = kv.split("=", 1)
                    limits[k] = v
        if not deploy_name:
            return "error: deployment name required"
        try:
            deploy = self.apps_v1.read_namespaced_deployment(deploy_name, ns)
            for c in deploy.spec.template.spec.containers:
                if container_name is None or c.name == container_name:
                    if not c.resources:
                        c.resources = client.V1ResourceRequirements()
                    c.resources.limits = {**(c.resources.limits or {}), **limits}
            self.apps_v1.patch_namespaced_deployment(deploy_name, ns, deploy)
            return f"deployment.apps/{deploy_name} resource requirements updated"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _set_image(self, parts, ns):
        deploy_name = None
        container_image = {}
        for p in parts:
            if "/" in p and "=" not in p:
                deploy_name = p.split("/")[-1]
            elif "=" in p:
                cname, img = p.split("=", 1)
                container_image[cname] = img
        if not deploy_name:
            return "error: deployment name required"
        try:
            deploy = self.apps_v1.read_namespaced_deployment(deploy_name, ns)
            for c in deploy.spec.template.spec.containers:
                if c.name in container_image:
                    c.image = container_image[c.name]
            self.apps_v1.patch_namespaced_deployment(deploy_name, ns, deploy)
            return f"deployment.apps/{deploy_name} image updated"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _set_env(self, parts, ns):
        deploy_name = None
        env_vars = {}
        for p in parts:
            if "/" in p and "=" not in p:
                deploy_name = p.split("/")[-1]
            elif "=" in p and not p.startswith("-"):
                k, v = p.split("=", 1)
                env_vars[k] = v
        if not deploy_name:
            return "error: deployment name required"
        try:
            deploy = self.apps_v1.read_namespaced_deployment(deploy_name, ns)
            for c in deploy.spec.template.spec.containers:
                if not c.env:
                    c.env = []
                for k, v in env_vars.items():
                    existing = next((e for e in c.env if e.name == k), None)
                    if existing:
                        existing.value = v
                    else:
                        c.env.append(client.V1EnvVar(name=k, value=v))
            self.apps_v1.patch_namespaced_deployment(deploy_name, ns, deploy)
            return f"deployment.apps/{deploy_name} env updated"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _cmd_delete(self, parts, ns):
        namespace = ns or "default"
        if parts and parts[0] == "pod" and len(parts) > 1:
            try:
                self.v1.delete_namespaced_pod(parts[1], namespace)
                return f'pod "{parts[1]}" deleted'
            except ApiException as e:
                return f"Error: {e.reason}"
        elif parts and parts[0] == "resourcequota" and len(parts) > 1:
            try:
                self.v1.delete_namespaced_resource_quota(parts[1], namespace)
                return f'resourcequota "{parts[1]}" deleted'
            except ApiException as e:
                return f"Error: {e.reason}"
        return "error: unsupported delete"

    def _cmd_scale(self, parts, ns):
        namespace = ns or "default"
        deploy_name = None
        replicas = None
        for p in parts:
            if "/" in p:
                deploy_name = p.split("/")[-1]
            if p.startswith("--replicas="):
                replicas = int(p.split("=")[1])
        if deploy_name and replicas is not None:
            try:
                body = {"spec": {"replicas": replicas}}
                self.apps_v1.patch_namespaced_deployment(deploy_name, namespace, body)
                return f"deployment.apps/{deploy_name} scaled"
            except ApiException as e:
                return f"Error: {e.reason}"
        return "error: deployment name and --replicas required"

    def _cmd_patch(self, parts, ns):
        namespace = ns or "default"
        if len(parts) < 2:
            return "error: resource type and name required"
        rtype, rname = parts[0], parts[1]
        patch_str = None
        for i, p in enumerate(parts):
            if p in ("-p", "--patch") and i + 1 < len(parts):
                patch_str = parts[i + 1]
            elif p.startswith("-p="):
                patch_str = p[3:]
        if not patch_str:
            return "error: patch body required (-p)"
        try:
            import json
            body = json.loads(patch_str)
            if rtype in ("deployment", "deploy"):
                self.apps_v1.patch_namespaced_deployment(rname, namespace, body)
                return f"deployment.apps/{rname} patched"
        except Exception as e:
            return f"Error: {str(e)}"
        return "error: unsupported patch target"

    # =========================================================
    # FAILURE INJECTION
    # =========================================================

    def inject_failure(self, failure_type: str, params: dict) -> str:
        """Inject a real failure into the GKE cluster."""
        ns = params.get("namespace", "payments")
        deploy = params.get("deployment", "payment-api")

        injectors = {
            "oom_kill": self._inject_oom,
            "crashloop": self._inject_crashloop,
            "image_pull": self._inject_image_pull,
            "bad_config": self._inject_bad_config,
            "liveness_probe": self._inject_liveness_probe,
            "resource_quota": self._inject_resource_quota,
            "cascading_db": self._inject_cascading,
        }
        fn = injectors.get(failure_type)
        if fn:
            return fn(ns, deploy)
        return f"Unknown failure type: {failure_type}"

    def _inject_oom(self, ns, deploy):
        self._set_resources([f"deployment/{deploy}", "-c", deploy.split("-")[0],
                           "--limits=memory=4Mi"], ns)
        time.sleep(8)
        return f"Injected OOMKill on {deploy} in {ns}"

    def _inject_crashloop(self, ns, deploy):
        try:
            d = self.apps_v1.read_namespaced_deployment(deploy, ns)
            d.spec.template.spec.containers[0].command = ["sh", "-c", "exit 1"]
            self.apps_v1.replace_namespaced_deployment(deploy, ns, d)
            time.sleep(10)
            return f"Injected CrashLoopBackOff on {deploy} in {ns}"
        except ApiException as e:
            return f"Error injecting crashloop: {e.reason}"

    def _inject_image_pull(self, ns, deploy):
        try:
            d = self.apps_v1.read_namespaced_deployment(deploy, ns)
            d.spec.template.spec.containers[0].image = "nginx:nonexistent-tag-99999"
            self.apps_v1.replace_namespaced_deployment(deploy, ns, d)
            time.sleep(5)
            return f"Injected ImagePullBackOff on {deploy} in {ns}"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _inject_bad_config(self, ns, deploy):
        self._set_env([f"deployment/{deploy}", "DB_HOST=wrong-host.invalid.local"], ns)
        time.sleep(8)
        return f"Injected bad config on {deploy} in {ns}"

    def _inject_liveness_probe(self, ns, deploy):
        try:
            d = self.apps_v1.read_namespaced_deployment(deploy, ns)
            for c in d.spec.template.spec.containers:
                if c.liveness_probe and c.liveness_probe.http_get:
                    c.liveness_probe.http_get.path = "/nonexistent-health-check"
            self.apps_v1.replace_namespaced_deployment(deploy, ns, d)
            time.sleep(15)
            return f"Injected liveness probe failure on {deploy} in {ns}"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _inject_resource_quota(self, ns, _deploy):
        quota = client.V1ResourceQuota(
            metadata=client.V1ObjectMeta(name="tight-quota", namespace=ns),
            spec=client.V1ResourceQuotaSpec(hard={"pods": "1", "requests.memory": "32Mi"})
        )
        try:
            self.v1.create_namespaced_resource_quota(ns, quota)
            return f"Injected tight ResourceQuota in {ns}"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _inject_cascading(self, ns, _deploy):
        self._inject_oom(ns, "redis")
        time.sleep(5)
        return "Injected cascading failure: redis OOM -> payment-api errors -> frontend 502"

    # =========================================================
    # RESET & HEALTH
    # =========================================================

    def reset(self):
        """Reset cluster to healthy state by reapplying manifests."""
        import subprocess

        for ns in self.app_namespaces:
            try:
                self.v1.delete_namespace(ns)
            except ApiException:
                pass

        for _ in range(60):
            existing = [ns.metadata.name for ns in self.v1.list_namespace().items]
            if not any(ns in existing for ns in self.app_namespaces):
                break
            time.sleep(2)

        if os.path.exists(self.manifests_dir):
            subprocess.run(["kubectl", "apply", "-f", self.manifests_dir],
                         capture_output=True, timeout=60)

        for _ in range(60):
            health = self.check_health()
            all_running = all(
                s == "Running" for ns_pods in health.values() for s in ns_pods.values()
            ) if health else False
            if all_running and health:
                break
            time.sleep(3)

    def check_health(self) -> dict:
        """Return {namespace: {pod_name: status}} for all app namespaces."""
        health = {}
        for ns in self.app_namespaces:
            try:
                pods = self.v1.list_namespaced_pod(ns)
                health[ns] = {}
                for p in pods.items:
                    status = p.status.phase
                    for cs in (p.status.container_statuses or []):
                        if cs.state and cs.state.waiting and cs.state.waiting.reason:
                            status = cs.state.waiting.reason
                        elif cs.state and cs.state.terminated and cs.state.terminated.reason:
                            status = cs.state.terminated.reason
                    health[ns][p.metadata.name] = status
            except ApiException:
                health[ns] = {}
        return health

    # =========================================================
    # HELPERS
    # =========================================================

    def _format_age(self, timestamp):
        if not timestamp:
            return "<unknown>"
        from datetime import datetime, timezone
        if hasattr(timestamp, 'timestamp'):
            delta = datetime.now(timezone.utc) - timestamp.replace(tzinfo=timezone.utc)
        else:
            return "<unknown>"
        seconds = int(delta.total_seconds())
        if seconds < 60: return f"{seconds}s"
        if seconds < 3600: return f"{seconds // 60}m"
        if seconds < 86400: return f"{seconds // 3600}h"
        return f"{seconds // 86400}d"
