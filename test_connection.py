#!/usr/bin/env python3
"""
Quick diagnostic: tests each component that could cause WebSocket failure.

Run on the same machine as the server:
  python test_connection.py

Or test just the client connection:
  python test_connection.py --client-only --url http://localhost:8000
"""

import argparse
import sys


def test_imports():
    """Test that all server imports work."""
    print("1. Testing imports...")
    try:
        from server.constants import TOPOLOGY, HEALTHY_STATE, APP_NAMESPACES
        print(f"   constants: OK (namespaces={APP_NAMESPACES})")
    except Exception as e:
        print(f"   constants: FAIL — {e}")
        return False

    try:
        from server.llm_client import LLMClient
        print("   llm_client: OK")
    except Exception as e:
        print(f"   llm_client: FAIL — {e}")
        return False

    try:
        from server.k8s_backend import K8sBackend
        print("   k8s_backend: OK")
    except Exception as e:
        print(f"   k8s_backend: FAIL — {e}")
        return False

    try:
        from server.kube_sre_gym_environment import KubeSreGymEnvironment
        print("   environment: OK")
    except Exception as e:
        print(f"   environment: FAIL — {e}")
        return False

    return True


def test_llm():
    """Test LLM client initialization."""
    import os
    print("\n2. Testing LLM client...")
    backend = os.environ.get("LLM_BACKEND", "openai")
    print(f"   LLM_BACKEND={backend}")
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        print(f"   ANTHROPIC_API_KEY={'set (' + key[:10] + '...)' if key else 'NOT SET'}")
        if not key:
            print("   FAIL: ANTHROPIC_API_KEY required for anthropic backend")
            return False
    try:
        from server.llm_client import LLMClient
        llm = LLMClient()
        print(f"   LLMClient({llm.backend}, {llm.model}): OK")
        return True
    except Exception as e:
        print(f"   LLMClient: FAIL — {e}")
        return False


def test_k8s():
    """Test K8s backend connectivity."""
    import os
    print("\n3. Testing K8s backend...")
    print(f"   K8S_ENDPOINT={'set' if os.environ.get('K8S_ENDPOINT') else 'NOT SET'}")
    print(f"   K8S_TOKEN={'set' if os.environ.get('K8S_TOKEN') else 'NOT SET'}")
    print(f"   K8S_CA_CERT={'set' if os.environ.get('K8S_CA_CERT') else 'NOT SET'}")
    try:
        from server.k8s_backend import K8sBackend
        backend = K8sBackend()
        print("   K8sBackend(): OK")
        health = backend.check_health()
        total_pods = sum(len(v) for v in health.values())
        print(f"   check_health(): {total_pods} pods across {len(health)} namespaces")
        for ns, pods in health.items():
            for pod, status in pods.items():
                print(f"     {ns}/{pod}: {status}")
        return True
    except Exception as e:
        print(f"   K8sBackend: FAIL — {e}")
        return False


def test_env_init():
    """Test full environment initialization."""
    print("\n4. Testing KubeSreGymEnvironment.__init__()...")
    try:
        from server.kube_sre_gym_environment import KubeSreGymEnvironment
        env = KubeSreGymEnvironment()
        print(f"   __init__(): OK (mode={env.mode})")
        return True
    except Exception as e:
        print(f"   __init__(): FAIL — {e}")
        return False


def test_client(url: str):
    """Test WebSocket client connection."""
    print(f"\n5. Testing client connection to {url}...")
    try:
        from kube_sre_gym import KubeSreGymEnv
        env = KubeSreGymEnv(base_url=url)
        print("   KubeSreGymEnv(): OK")
        print("   Calling reset()...")
        result = env.reset()
        print(f"   reset(): OK — {result.observation.command_output[:100]}...")
        env.close()
        return True
    except Exception as e:
        print(f"   FAIL — {type(e).__name__}: {e}")
        return False


def test_http_health(url: str):
    """Test the /healthz endpoint."""
    print(f"\n5a. Testing HTTP health check at {url}/healthz...")
    try:
        import requests
        resp = requests.get(f"{url}/healthz", timeout=30)
        data = resp.json()
        print(f"   Status: {data.get('status')}")
        if data.get("error"):
            print(f"   Error: {data['error']}")
        return data.get("status") == "ok"
    except Exception as e:
        print(f"   FAIL — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-only", action="store_true",
                        help="Only test client connection (skip server-side checks)")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    if args.client_only:
        ok = test_http_health(args.url)
        if ok:
            test_client(args.url)
        return

    all_ok = True
    all_ok &= test_imports()
    all_ok &= test_llm()
    all_ok &= test_k8s()
    all_ok &= test_env_init()

    if all_ok:
        print("\nAll server-side checks passed. Try client connection:")
        print(f"  python test_connection.py --client-only --url {args.url}")
    else:
        print("\nSome checks failed. Fix the issues above before starting the server.")
        sys.exit(1)


if __name__ == "__main__":
    main()
