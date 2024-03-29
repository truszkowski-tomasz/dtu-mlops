# Run docker image with the following command:
# docker run -e WANDB_API_KEY=<API_KEY> trainer:latest
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

WORKDIR /
COPY src/ src/

COPY models/ models/
COPY data/ data/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN pip install -e .

ENTRYPOINT ["python", "-u", "src/train_model.py"]
