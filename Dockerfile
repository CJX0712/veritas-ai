FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY golden ./golden
RUN pip install --no-cache-dir . && pip install --no-cache-dir "uvicorn>=0.53,<0.54"

EXPOSE 8000
ENV VERITAS_TRACK=offline
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "veritas.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Author: 晨星
