FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

EXPOSE <port>

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app/main.py", "--server.port=<port>", "--server.address=<ip_address>"]