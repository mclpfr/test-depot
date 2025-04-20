FROM python:3.10-slim-buster

# Arguments pour UID et GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

COPY requirements.txt .

ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models /app/data && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data /app/models

COPY config.yaml .

COPY src/train_model/train_model.py .

USER appuser

CMD ["python", "train_model.py"]
