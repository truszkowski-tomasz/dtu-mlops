# Run docker image with the following command:
# docker run -e WANDB_API_KEY=<API_KEY> trainer:latest
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY data.dvc data.dvc
COPY models.dvc models.dvc

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

COPY src/ src/
RUN pip install -e .

COPY .dvc/ .dvc/
RUN dvc pull

COPY models/ models/
COPY data/ data/

RUN pip install . --no-deps --no-cache-dir


ENTRYPOINT ["python", "-u", "src/train_model.py"]
