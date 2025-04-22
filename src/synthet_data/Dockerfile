FROM python:3.10-slim-buster

# Arguments pour UID et GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Configuration des limites de mémoire et d'optimisation
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=384m

RUN pip install --no-cache-dir pandas numpy pyyaml scikit-learn sdv

RUN mkdir -p /app/data/raw && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data

COPY config.yaml .
COPY src/synthet_data/synthet_data.py .

USER appuser

CMD ["python", "synthet_data.py"] 