FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install uv && \
    uv pip install -r requirements.txt

EXPOSE 8000

CMD ["uv", "run", "python", "serve.py"]