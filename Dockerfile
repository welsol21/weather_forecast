FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY hly4935_subset.csv ./hly4935_subset.csv
COPY TZ.docx ./TZ.docx
COPY prompt.txt ./prompt.txt

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "weather_patterns", "run-pipeline", "--csv", "hly4935_subset.csv", "--output-dir", "artifacts"]
