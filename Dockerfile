FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "k8s_sre_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
