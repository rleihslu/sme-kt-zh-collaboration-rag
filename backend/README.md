




## Docker:
<!-- FRom backend
``
docker buildx build --platform linux/amd64,linux/arm64 -t pkoerner6/renku-ollama-baseline:v2 --push .
`` -->
From source:
``
docker buildx build --platform linux/amd64,linux/arm64 -f backend/Dockerfile -t pkoerner6/renku-ollama-baseline:v2 --push .
``
