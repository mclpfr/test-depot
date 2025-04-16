FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && PIP_PROGRESS_BAR=off pip install --no-cache-dir -r requirements.txt

COPY src/auto_dvc/. .

CMD ["python", "auto_dvc.py"] 