FROM python:3.11-slim

WORKDIR /app

# 先装依赖, 利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fusionrag ./fusionrag
COPY webui ./webui
COPY pytest.ini .

ENV WORKING_DIR=/app/fusionrag_workspace \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

VOLUME ["/app/fusionrag_workspace"]

CMD ["python", "-m", "fusionrag"]
