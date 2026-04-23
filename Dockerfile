FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python training/generate_data.py

EXPOSE 7860
CMD ["uvicorn", "environment.server:app", "--host", "0.0.0.0", "--port", "7860"]
