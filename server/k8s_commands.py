"""
Kubectl command handlers — parses and executes kubectl-style commands via K8s API.

Dict-based dispatch replaces the if/elif chain. Each handler is a method that
takes (parts, namespace) and returns a string (matching kubectl output format).
"""

import logging
import re
import time
from datetime import datetime, timezone

from kubernetes import client
from kubernetes.client.rest import ApiException
from tabulate import tabulate

from .constants import DEFAULT_NAMESPACE

logger = logging.getLogger(__name__)


class CommandHandler:
    """Executes kubectl-style commands against the K8s API."""

    def __init__(self, v1: client.CoreV1Api, apps_v1: client.AppsV1Api):
        self.v1 = v1
        self.apps_v1 = apps_v1

    def dispatch(self, verb: str, parts: list[str], ns: str | None, raw_cmd: str = "") -> str:
        """Route a kubectl verb to its handler. Returns kubectl-style output."""
        commands = {
            "get": self._cmd_get,
            "describe": self._cmd_describe,
            "logs": self._cmd_logs,
            "top": self._cmd_top,
            "rollout": self._cmd_rollout,
            "set": self._cmd_set,
            "delete": self._cmd_delete,
            "scale": self._cmd_scale,
        }
        if verb == "patch":
            return self._cmd_patch(raw_cmd, ns)

        handler = commands.get(verb)
        if not handler:
            return f"error: unknown command '{verb}'"
        return handler(parts, ns)

    # ---- get ----

    _GET_RESOURCES = None  # built lazily

    def _cmd_get(self, parts: list[str], ns: str | None) -> str:
        if not parts:
            return "error: resource type required"

        if self._GET_RESOURCES is None:
            self._GET_RESOURCES = {
                "pods": self._get_pods, "pod": self._get_pods, "po": self._get_pods,
                "deployments": self._get_deployments, "deployment": self._get_deployments, "deploy": self._get_deployments,
                "events": self._get_events, "ev": self._get_events,
                "nodes": self._get_nodes, "node": self._get_nodes,
                "services": self._get_services, "svc": self._get_services,
                "resourcequota": self._get_resourcequotas, "quota": self._get_resourcequotas,
            }

        resource = parts[0]
        handler = self._GET_RESOURCES.get(resource)
        if not handler:
            return f'error: the server doesn\'t have a resource type "{resource}"'
        return handler(ns)

    def _get_pods(self, ns: str | None) -> str:
        if ns == "__all__" or not ns:
            pods = self.v1.list_pod_for_all_namespaces()
        else:
            pods = self.v1.list_namespaced_pod(ns)

        if not pods.items:
            return "No resources found."

        rows = []
        for p in pods.items:
            total = len(p.spec.containers)
            ready_count = sum(1 for cs in (p.status.container_statuses or []) if cs.ready)
            status = _pod_status(p)
            restarts = sum(cs.restart_count for cs in (p.status.container_statuses or []))
            rows.append([
                p.metadata.name,
                f"{ready_count}/{total}",
                status,
                str(restarts),
                _format_age(p.metadata.creation_timestamp),
            ])
        return tabulate(rows, headers=["NAME", "READY", "STATUS", "RESTARTS", "AGE"], tablefmt="plain")

    def _get_deployments(self, ns: str | None) -> str:
        if ns and ns != "__all__":
            deploys = self.apps_v1.list_namespaced_deployment(ns)
        else:
            deploys = self.apps_v1.list_deployment_for_all_namespaces()

        rows = []
        for d in deploys.items:
            rows.append([
                d.metadata.name,
                f"{d.status.ready_replicas or 0}/{d.spec.replicas or 0}",
                str(d.status.updated_replicas or 0),
                str(d.status.available_replicas or 0),
                _format_age(d.metadata.creation_timestamp),
            ])
        return tabulate(rows, headers=["NAME", "READY", "UP-TO-DATE", "AVAILABLE", "AGE"], tablefmt="plain")

    def _get_events(self, ns: str | None) -> str:
        if ns and ns != "__all__":
            events = self.v1.list_namespaced_event(ns)
        else:
            events = self.v1.list_event_for_all_namespaces()

        # Sort by time; key is always str so we never mix None with datetime (avoids TypeError)
        def _key(e):
            t = e.last_timestamp or e.metadata.creation_timestamp
            return getattr(t, "isoformat", lambda: "")() if t else "z"

        sorted_events = sorted(events.items, key=_key)[-20:]
        rows = []
        for e in sorted_events:
            rows.append([
                _format_age(e.last_timestamp or e.metadata.creation_timestamp),
                e.type or "Normal",
                e.reason or "",
                f"{e.involved_object.kind}/{e.involved_object.name}"[:20],
                (e.message or "")[:80],
            ])
        return tabulate(rows, headers=["LAST SEEN", "TYPE", "REASON", "OBJECT", "MESSAGE"], tablefmt="plain")

    def _get_nodes(self, _ns: str | None = None) -> str:
        nodes = self.v1.list_node()
        rows = []
        for n in nodes.items:
            conditions = {c.type: c.status for c in (n.status.conditions or [])}
            status = "Ready" if conditions.get("Ready") == "True" else "NotReady"
            labels = n.metadata.labels or {}
            roles = ",".join(k.split("/")[-1] for k in labels if "node-role" in k) or "<none>"
            version = getattr(n.status.node_info, "kubelet_version", "") if n.status.node_info else ""
            rows.append([
                n.metadata.name,
                status,
                roles,
                _format_age(n.metadata.creation_timestamp),
                version,
            ])
        return tabulate(rows, headers=["NAME", "STATUS", "ROLES", "AGE", "VERSION"], tablefmt="plain")

    def _get_services(self, ns: str | None) -> str:
        if ns and ns != "__all__":
            svcs = self.v1.list_namespaced_service(ns)
        else:
            svcs = self.v1.list_service_for_all_namespaces()

        rows = []
        for s in svcs.items:
            ports = ",".join(f"{p.port}/{p.protocol}" for p in (s.spec.ports or []))
            rows.append([
                s.metadata.name,
                s.spec.type or "ClusterIP",
                s.spec.cluster_ip or "None",
                ports,
            ])
        return tabulate(rows, headers=["NAME", "TYPE", "CLUSTER-IP", "PORT(S)"], tablefmt="plain")

    def _get_resourcequotas(self, ns: str | None) -> str:
        if not ns or ns == "__all__":
            return "error: namespace required for resourcequota"
        quotas = self.v1.list_namespaced_resource_quota(ns)
        if not quotas.items:
            return "No resources found."

        lines = []
        for q in quotas.items:
            lines.append(f"Name:    {q.metadata.name}")
            rows = []
            for resource, hard in (q.status.hard or {}).items():
                used = (q.status.used or {}).get(resource, "0")
                rows.append([resource, str(used), hard])
            lines.append(tabulate(rows, headers=["Resource", "Used", "Hard"], tablefmt="plain"))
        return "\n".join(lines)

    # ---- describe ----

    def _cmd_describe(self, parts: list[str], ns: str | None) -> str:
        if len(parts) < 2:
            return "error: resource name required"

        describe_map = {
            "pod": self._describe_pod, "pods": self._describe_pod, "po": self._describe_pod,
            "deployment": self._describe_deployment, "deploy": self._describe_deployment,
            "node": self._describe_node, "nodes": self._describe_node,
        }
        rtype, rname = parts[0], parts[1]
        handler = describe_map.get(rtype)
        if not handler:
            return f"error: unsupported describe for {rtype}"
        return handler(rname, ns)

    def _describe_pod(self, name: str, ns: str | None) -> str:
        namespace = ns or DEFAULT_NAMESPACE
        pods = self.v1.list_namespaced_pod(namespace)
        pod = next((p for p in pods.items if name in p.metadata.name), None)
        if not pod:
            return f'Error from server (NotFound): pods "{name}" not found in namespace "{namespace}"'

        lines = [
            f"Name:         {pod.metadata.name}",
            f"Namespace:    {pod.metadata.namespace}",
            f"Node:         {pod.spec.node_name or '<none>'}",
            f"Status:       {pod.status.phase}",
            f"IP:           {pod.status.pod_ip or '<none>'}",
            "Containers:",
        ]
        for c in pod.spec.containers:
            lines.append(f"  {c.name}:")
            lines.append(f"    Image:          {c.image}")
            cs = next((s for s in (pod.status.container_statuses or []) if s.name == c.name), None)
            if cs:
                lines.extend(_format_container_status(cs))
            if c.resources:
                if c.resources.limits:
                    lines.append("    Limits:")
                    for k, v in c.resources.limits.items():
                        lines.append(f"      {k}:     {v}")
                if c.resources.requests:
                    lines.append("    Requests:")
                    for k, v in c.resources.requests.items():
                        lines.append(f"      {k}:     {v}")
            if c.liveness_probe:
                lines.append(f"    Liveness:       {_format_probe(c.liveness_probe)}")
            if c.env:
                lines.append("    Environment:")
                for e in c.env:
                    lines.append(f"      {e.name}: {e.value}")

        events = self.v1.list_namespaced_event(
            namespace, field_selector=f"involvedObject.name={pod.metadata.name}")
        if events.items:
            lines.append("Events:")
            for e in events.items[-10:]:
                lines.append(f"  {e.type}\t{e.reason}\t{e.message}")

        return "\n".join(lines)

    def _describe_deployment(self, name: str, ns: str | None) -> str:
        namespace = ns or DEFAULT_NAMESPACE
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
        for cond in (d.status.conditions or []):
            lines.append(f"  {cond.type}: {cond.status} ({cond.reason}) - {cond.message}")

        lines.append("Pod Template:")
        for c in d.spec.template.spec.containers:
            lines.append(f"  Container: {c.name}")
            lines.append(f"    Image:      {c.image}")
            if c.command:
                lines.append(f"    Command:    {c.command}")
            if c.resources and c.resources.limits:
                lines.append(f"    Limits:     {', '.join(f'{k}={v}' for k, v in c.resources.limits.items())}")
            if c.resources and c.resources.requests:
                lines.append(f"    Requests:   {', '.join(f'{k}={v}' for k, v in c.resources.requests.items())}")
            if c.liveness_probe:
                lines.append(f"    Liveness:   {_format_probe(c.liveness_probe)}")
            if c.readiness_probe:
                lines.append(f"    Readiness:  {_format_probe(c.readiness_probe)}")
            if c.env:
                lines.append("    Environment:")
                for e in c.env:
                    lines.append(f"      {e.name}: {e.value}")

        events = self.v1.list_namespaced_event(
            namespace, field_selector=f"involvedObject.name={d.metadata.name}")
        if events.items:
            lines.append("Events:")
            for e in events.items[-5:]:
                lines.append(f"  {e.type}\t{e.reason}\t{e.message}")

        return "\n".join(lines)

    def _describe_node(self, name: str, _ns: str | None = None) -> str:
        try:
            n = self.v1.read_node(name)
        except ApiException:
            return f'Error from server (NotFound): nodes "{name}" not found'
        alloc = n.status.allocatable or {}
        return (f"Name:         {n.metadata.name}\n"
                f"Allocatable:  cpu={alloc.get('cpu', '?')}, "
                f"memory={alloc.get('memory', '?')}")

    # ---- logs ----

    def _cmd_logs(self, parts: list[str], ns: str | None) -> str:
        if not parts:
            return "error: pod name required"
        pod_name = parts[0]
        namespace = ns or DEFAULT_NAMESPACE
        tail = 50
        container = None
        previous = False
        for i, p in enumerate(parts):
            if p.startswith("--tail="):
                try:
                    tail = int(p.split("=")[1])
                except (ValueError, IndexError):
                    pass
            if p == "-c" and i + 1 < len(parts):
                container = parts[i + 1]
            if p in ("--previous", "-p"):
                previous = True

        pods = self.v1.list_namespaced_pod(namespace)
        matched = next((p for p in pods.items if pod_name in p.metadata.name), None)
        if not matched:
            return f'Error from server (NotFound): pods "{pod_name}" not found'
        try:
            logs = self.v1.read_namespaced_pod_log(
                matched.metadata.name, namespace, container=container,
                tail_lines=tail, previous=previous)
            return logs if logs else "(no logs)"
        except ApiException as e:
            return f"Error: {e.reason}"

    # ---- top ----

    def _cmd_top(self, parts: list[str], ns: str | None) -> str:
        if not parts:
            return "error: resource type required"
        if parts[0] == "pods":
            return self._top_pods(ns)
        elif parts[0] == "nodes":
            return self._top_nodes()
        return "error: unsupported"

    def _top_pods(self, ns: str | None) -> str:
        try:
            api = client.CustomObjectsApi()
            if ns and ns != "__all__":
                metrics = api.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", ns, "pods")
            else:
                metrics = api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "pods")
            rows = []
            for item in metrics.get("items", []):
                for c in item.get("containers", []):
                    rows.append([
                        item["metadata"]["name"],
                        c["usage"].get("cpu", "0"),
                        c["usage"].get("memory", "0"),
                    ])
            return tabulate(rows, headers=["NAME", "CPU(cores)", "MEMORY(bytes)"], tablefmt="plain")
        except Exception:
            return "error: Metrics API not available. Use 'kubectl describe pod' to see resource requests/limits."

    def _top_nodes(self) -> str:
        try:
            api = client.CustomObjectsApi()
            metrics = api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")
            rows = []
            for item in metrics.get("items", []):
                rows.append([
                    item["metadata"]["name"],
                    item["usage"].get("cpu", "0"),
                    "-",
                    item["usage"].get("memory", "0"),
                    "-",
                ])
            return tabulate(rows, headers=["NAME", "CPU(cores)", "CPU%", "MEMORY(bytes)", "MEMORY%"], tablefmt="plain")
        except Exception:
            return "error: Metrics API not available."

    # ---- mutation commands (also used by injectors) ----

    def rollout_restart(self, deploy_name: str, namespace: str) -> str:
        """Restart a deployment (triggers new rollout)."""
        try:
            body = {"spec": {"template": {"metadata": {"annotations": {
                "kubectl.kubernetes.io/restartedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }}}}}
            self.apps_v1.patch_namespaced_deployment(deploy_name, namespace, body)
            return f"deployment.apps/{deploy_name} restarted"
        except ApiException as e:
            return f"Error: {e.reason}"

    def _cmd_rollout(self, parts: list[str], ns: str | None) -> str:
        if parts and parts[0] == "restart" and len(parts) > 1:
            deploy_name = parts[1].replace("deployment/", "")
            return self.rollout_restart(deploy_name, ns or DEFAULT_NAMESPACE)
        return "error: unsupported rollout command"

    def set_resources(self, parts: list[str], ns: str) -> str:
        """Set resource limits on a deployment. Used by both agent commands and injectors."""
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
            matched = False
            available_containers = [c.name for c in deploy.spec.template.spec.containers]
            for c in deploy.spec.template.spec.containers:
                if container_name is None or c.name == container_name:
                    if not c.resources:
                        c.resources = client.V1ResourceRequirements()
                    c.resources.limits = {**(c.resources.limits or {}), **limits}
                    matched = True
            if not matched:
                return (f"error: container '{container_name}' not found in deployment {deploy_name}. "
                        f"Available containers: {available_containers}")
            self.apps_v1.patch_namespaced_deployment(deploy_name, ns, deploy)
            return f"deployment.apps/{deploy_name} resource requirements updated"
        except ApiException as e:
            return f"Error: {e.reason}"

    def set_image(self, parts: list[str], ns: str) -> str:
        """Set container image on a deployment."""
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
            matched = False
            available_containers = [c.name for c in deploy.spec.template.spec.containers]
            for c in deploy.spec.template.spec.containers:
                if c.name in container_image:
                    c.image = container_image[c.name]
                    matched = True
            if not matched:
                requested = list(container_image.keys())
                return (f"error: container(s) {requested} not found in deployment {deploy_name}. "
                        f"Available containers: {available_containers}")
            self.apps_v1.patch_namespaced_deployment(deploy_name, ns, deploy)
            return f"deployment.apps/{deploy_name} image updated"
        except ApiException as e:
            return f"Error: {e.reason}"

    def set_env(self, parts: list[str], ns: str) -> str:
        """Set environment variables on a deployment."""
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

    def _cmd_set(self, parts: list[str], ns: str | None) -> str:
        namespace = ns or DEFAULT_NAMESPACE
        if not parts:
            return "error: subcommand required"
        sub = parts[0]
        if sub == "resources":
            return self.set_resources(parts[1:], namespace)
        elif sub == "image":
            return self.set_image(parts[1:], namespace)
        elif sub == "env":
            return self.set_env(parts[1:], namespace)
        return "error: unsupported set command"

    # ---- delete ----

    def _cmd_delete(self, parts: list[str], ns: str | None) -> str:
        namespace = ns or DEFAULT_NAMESPACE
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

    # ---- scale ----

    def _cmd_scale(self, parts: list[str], ns: str | None) -> str:
        namespace = ns or DEFAULT_NAMESPACE
        deploy_name = None
        replicas = None
        for p in parts:
            if "/" in p:
                deploy_name = p.split("/")[-1]
            if p.startswith("--replicas="):
                try:
                    replicas = int(p.split("=")[1])
                except (ValueError, IndexError):
                    return "error: --replicas must be an integer"
        if deploy_name and replicas is not None:
            try:
                body = {"spec": {"replicas": replicas}}
                self.apps_v1.patch_namespaced_deployment(deploy_name, namespace, body)
                return f"deployment.apps/{deploy_name} scaled"
            except ApiException as e:
                return f"Error: {e.reason}"
        return "error: deployment name and --replicas required"

    # ---- patch ----

    def _cmd_patch(self, raw_cmd: str, ns: str | None) -> str:
        """Handle kubectl patch — parses from raw command string to preserve JSON body."""
        import json
        namespace = ns or DEFAULT_NAMESPACE
        parts = raw_cmd.split()
        if len(parts) < 3:
            return "error: resource type and name required"
        rtype, rname = parts[1], parts[2]

        # Extract JSON patch body — find first '{' and match braces
        # Strip all single quotes first (shell quoting artifacts)
        cleaned_cmd = raw_cmd.replace("'", "")
        brace_start = cleaned_cmd.find("{")
        if brace_start < 0:
            # Also try finding JSON after -p flag
            return "error: patch body required (-p '{...}')"

        patch_str = None
        depth = 0
        for j, ch in enumerate(cleaned_cmd[brace_start:], brace_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    patch_str = cleaned_cmd[brace_start:j + 1]
                    break

        if not patch_str:
            return "error: patch body required (-p '{...}')"
        try:
            body = json.loads(patch_str)
            if rtype in ("deployment", "deploy", "deployments"):
                self.apps_v1.patch_namespaced_deployment(rname, namespace, body)
                return f"deployment.apps/{rname} patched"
        except ValueError as e:
            return f"error: invalid JSON in patch body: {e}"
        except ApiException as e:
            return f"Error from server ({e.reason}): {e.body}"
        return "error: unsupported patch target"


# ---- Shared utility functions ----

def _format_age(timestamp) -> str:
    """Format a K8s timestamp into a human-readable age string."""
    if not timestamp:
        return "<unknown>"
    if not hasattr(timestamp, 'timestamp'):
        return "<unknown>"
    delta = datetime.now(timezone.utc) - timestamp.replace(tzinfo=timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _pod_status(pod) -> str:
    """Extract the display status for a pod (matches kubectl output)."""
    status = pod.status.phase or "Unknown"
    for cs in (pod.status.container_statuses or []):
        if cs.state and cs.state.waiting and cs.state.waiting.reason:
            status = cs.state.waiting.reason
        elif cs.state and cs.state.terminated and cs.state.terminated.reason:
            status = cs.state.terminated.reason
    return status


def _format_probe(probe) -> str:
    """Format a liveness/readiness probe for display."""
    if not probe:
        return "configured"
    if probe.http_get:
        p = probe.http_get
        return (f"http-get {p.path or '/'}:{p.port or 0} "
                f"delay={probe.initial_delay_seconds or 0}s period={probe.period_seconds or 0}s")
    if probe._exec and probe._exec.command:
        return (f"exec {probe._exec.command} "
                f"delay={probe.initial_delay_seconds}s period={probe.period_seconds}s")
    if probe.tcp_socket:
        return (f"tcp-socket :{probe.tcp_socket.port} "
                f"delay={probe.initial_delay_seconds}s period={probe.period_seconds}s")
    return "configured"


def _format_container_status(cs) -> list[str]:
    """Format container status lines for describe output."""
    lines = []
    if cs.state and cs.state.running:
        lines.append("    State:          Running")
    elif cs.state and cs.state.waiting:
        lines.append("    State:          Waiting")
        lines.append(f"      Reason:       {cs.state.waiting.reason}")
    elif cs.state and cs.state.terminated:
        lines.append("    State:          Terminated")
        lines.append(f"      Reason:       {cs.state.terminated.reason}")
        lines.append(f"      Exit Code:    {cs.state.terminated.exit_code}")
    if cs.last_state and cs.last_state.terminated:
        lines.append("    Last State:     Terminated")
        lines.append(f"      Reason:       {cs.last_state.terminated.reason}")
        lines.append(f"      Exit Code:    {cs.last_state.terminated.exit_code}")
    lines.append(f"    Ready:          {cs.ready}")
    lines.append(f"    Restart Count:  {cs.restart_count}")
    return lines
