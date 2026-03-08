"""
Shared constants for the Kube SRE Gym environment.

Central place for topology, failure descriptions, namespace lists, and defaults.
Imported by k8s_backend, scenario_generator, adversarial_designer, and environment.
"""

# ---- Cluster topology: namespace -> deployments ----
# Must match sample_app/base/ manifests
TOPOLOGY = {
    "payments": ["payment-gateway", "payment-db", "payment-worker", "payment-api"],
    "frontend": ["web-app", "frontend-cache"],
    "auth": ["auth-service", "auth-db"],
}

APP_NAMESPACES = ["payments", "frontend", "auth", "hackathon"]

DEFAULT_NAMESPACE = "default"
MAX_STEPS = 15

# ---- Injectable failure types (descriptions for LLM prompts) ----
INJECTABLE_FAILURES = {
    "oom_kill": "Sets memory limit to 4Mi — pod OOMKills (exit code 137)",
    "crashloop": "Changes container command to 'exit 1' — CrashLoopBackOff",
    "image_pull": "Sets image to nonexistent tag — ImagePullBackOff/ErrImagePull",
    "bad_config": "Sets DB_HOST env var to invalid host — app connection errors",
    "liveness_probe": "Sets liveness probe to wrong path — pod restart loop",
    "resource_quota": "Applies tight ResourceQuota — new pods blocked from scheduling",
    "cascading_db": "OOMs redis — payment-api loses backend — frontend 502s",
}

# ---- Healthy steady-state for each deployment ----
# Used by reset() to restore deployments, by injectors for container name lookup,
# and by adversarial_designer for context.
# Fields:
#   container_name  — first container's name (for set image/resources commands)
#   image           — correct image tag
#   env             — correct env vars
#   memory_limit    — correct memory limit
#   replicas        — correct replica count
#   liveness_probe  — {path, port} if deployment has one, else None
HEALTHY_STATE = {
    "payments": {
        "payment-gateway": {
            "container_name": "payment-gateway", "image": "nginx:1.25",
            "env": {}, "memory_limit": "128Mi", "replicas": 2,
            "liveness_probe": {"path": "/health", "port": 8080},
        },
        "payment-db": {
            "container_name": "payment-db", "image": "postgres:15",
            "env": {"POSTGRES_DB": "payments", "POSTGRES_USER": "payments_user"},
            "memory_limit": "512Mi", "replicas": 1,
            "liveness_probe": None,
        },
        "payment-worker": {
            "container_name": "payment-worker", "image": "busybox:1.36",
            "env": {"DATABASE_URL": "postgres://payment-db.payments.svc.cluster.local:5432/payments"},
            "memory_limit": "64Mi", "replicas": 1,
            "liveness_probe": None,
        },
        "payment-api": {
            "container_name": "payment-api", "image": "python:3.11-slim",
            "env": {"DB_HOST": "payment-db.payments.svc.cluster.local", "PORT": "8080"},
            "memory_limit": "64Mi", "replicas": 1,
            "liveness_probe": {"path": "/health", "port": 8080},
        },
    },
    "frontend": {
        "web-app": {
            "container_name": "web-app", "image": "nginx:1.25",
            "env": {"API_URL": "http://payment-api.payments.svc.cluster.local:8080", "PORT": "3000"},
            "memory_limit": "128Mi", "replicas": 2,
            "liveness_probe": {"path": "/health", "port": 3000},
        },
        "frontend-cache": {
            "container_name": "frontend-cache", "image": "redis:7",
            "env": {}, "memory_limit": "128Mi", "replicas": 1,
            "liveness_probe": None,
        },
    },
    "auth": {
        "auth-service": {
            "container_name": "auth-service", "image": "nginx:1.25",
            "env": {"DB_HOST": "auth-db.auth.svc.cluster.local", "TOKEN_SECRET": "supersecret", "PORT": "8081"},
            "memory_limit": "128Mi", "replicas": 2,
            "liveness_probe": {"path": "/health", "port": 8081},
        },
        "auth-db": {
            "container_name": "auth-db", "image": "postgres:15",
            "env": {"POSTGRES_DB": "auth", "POSTGRES_USER": "auth_user"},
            "memory_limit": "512Mi", "replicas": 1,
            "liveness_probe": None,
        },
    },
}

# ---- Timeouts ----
RESET_POLL_INTERVAL = 3     # seconds between health checks during reset
RESET_MAX_POLLS = 30        # max iterations waiting for healthy pods
INJECT_WAIT_DEFAULT = 8     # seconds to wait after fault injection
