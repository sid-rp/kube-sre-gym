#!/usr/bin/env bash
# deploy_all.sh — Deploy the kube-sre-gym sample app to a K8s cluster
#
# Usage:
#   ./sample_app/deploy_all.sh              # Deploy healthy base only (for training)
#   ./sample_app/deploy_all.sh --with-faults  # Deploy base + broken scenarios (for testing)
#
# The OpenEnv server calls reset() which deploys base/ automatically.
# This script is for manual setup/testing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_FAULTS=false

for arg in "$@"; do
    case "$arg" in
        --with-faults) WITH_FAULTS=true ;;
    esac
done

echo "========================================="
echo " kube-sre-gym: Deploying Sample App"
echo "========================================="

# Step 1: Create namespaces
echo ""
echo "[1/3] Creating namespaces..."
kubectl apply -f "$SCRIPT_DIR/namespaces.yaml"

# Step 2: Deploy healthy base apps
echo ""
echo "[2/3] Deploying healthy base apps..."
kubectl apply -R -f "$SCRIPT_DIR/base/"

if [ "$WITH_FAULTS" = true ]; then
    # Step 3: Also deploy broken scenarios (for testing, NOT for training)
    echo ""
    echo "[3/3] Deploying broken scenarios (--with-faults)..."
    for dir in payments frontend auth; do
        if [ -d "$SCRIPT_DIR/$dir" ]; then
            kubectl apply -f "$SCRIPT_DIR/$dir/"
        fi
    done
    for dir in hackathon/training hackathon/eval hackathon/complex; do
        if [ -d "$SCRIPT_DIR/$dir" ]; then
            kubectl apply -f "$SCRIPT_DIR/$dir/"
        fi
    done
    echo ""
    echo "WARNING: Broken scenarios deployed. This is for manual testing only."
    echo "For training, use just: ./sample_app/deploy_all.sh (no --with-faults)"
else
    echo ""
    echo "[3/3] Skipping broken scenarios (use --with-faults to include them)"
fi

echo ""
echo "========================================="
echo " Deployment complete!"
echo "========================================="
echo ""
echo "Base healthy apps: payment-gateway, payment-db, payment-worker, payment-api,"
echo "                   web-app, frontend-cache, auth-service, auth-db,"
echo "                   api-server, ml-worker, redis, payment-service"
echo ""
echo "Check status:  kubectl get pods -A | grep -E 'payments|frontend|auth|hackathon'"
echo ""
echo "The OpenEnv server will inject faults via curriculum-driven reset()."
echo "Faults are injected ONE at a time into the clean cluster for clear training signal."
