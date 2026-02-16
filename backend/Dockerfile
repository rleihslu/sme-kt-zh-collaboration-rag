FROM ghcr.io/swissdatasciencecenter/renku/py-basic-jupyterlab:2.13.2

USER root

RUN apt-get update && apt-get install -y zstd && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

USER renku

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# CMD ["python3 backend/src/sme_kt_zh_collaboration_rag/baseline_rag.py"]