#!/usr/bin/env bash
# cleanup.sh — Delete all kube-sre-gym sample app resources for reset between episodes
# Usage: ./sample_app/cleanup.sh
#
# Deletes all resources in payments, frontend, auth, and hackathon namespaces,
# then optionally deletes the namespaces themselves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo " kube-sre-gym: Cleaning Up Sample App"
echo "========================================="

NAMESPACES="payments frontend auth hackathon"

for ns in $NAMESPACES; do
    echo ""
    echo "Cleaning namespace: $ns"

    # Delete all deployments, statefulsets, services, pods, configmaps, networkpolicies
    kubectl delete deployments --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete statefulsets --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete services --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete pods --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete configmaps --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete networkpolicies --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    kubectl delete pvc --all -n "$ns" --ignore-not-found --wait=false 2>/dev/null || true
done

# Parse flags
DELETE_NS=false
for arg in "$@"; do
    case "$arg" in
        --delete-namespaces) DELETE_NS=true ;;
    esac
done

if [ "$DELETE_NS" = true ]; then
    echo ""
    echo "Deleting namespaces..."
    for ns in $NAMESPACES; do
        kubectl delete namespace "$ns" --ignore-not-found --wait=false 2>/dev/null || true
    done
fi

echo ""
echo "========================================="
echo " Cleanup complete!"
echo "========================================="
echo ""
echo "To redeploy:  ./sample_app/deploy_all.sh"
echo "To also delete namespaces:  ./sample_app/cleanup.sh --delete-namespaces"
