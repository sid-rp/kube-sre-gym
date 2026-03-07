FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl apt-transport-https ca-certificates gnupg \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin kubectl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN chown -R user:user /app
USER user

# HF Spaces expects port 7860
# Set these as HF Spaces Secrets:
#   GCP_SA_KEY_JSON    — raw JSON string of the service account key
#   GKE_CLUSTER_ENDPOINT — e.g. https://34.169.10.97
#   GKE_CA_CERT        — base64 CA cert from kubeconfig certificate-authority-data
#   LLM_BASE_URL       — vLLM server URL (e.g. http://<H100-ip>:8000/v1)
#   LLM_MODEL          — model name on vLLM server

EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
