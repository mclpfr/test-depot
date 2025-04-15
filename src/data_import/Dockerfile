FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/postgresql/wait-for-postgres.sh .
RUN chmod +x wait-for-postgres.sh

COPY src/postgresql/import_data.py .
COPY config.yaml .

CMD ["python", "import_data.py"] 
