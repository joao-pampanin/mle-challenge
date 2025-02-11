FROM python:3.11-slim

WORKDIR /bain-desafio-mle

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src/ src/
COPY data/ data/
