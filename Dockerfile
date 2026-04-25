FROM python:3.11-slim

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .
RUN python training/generate_data.py

EXPOSE 7860
CMD ["python", "demo/app.py"]
