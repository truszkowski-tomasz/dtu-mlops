# If you want to build an image yourself:
# docker build -f dockerfiles/predict_model.dockerfile . -t <image_name>

# Run the app with the following command:
# docker run --name <choose_your_container_name> -p 80:80 <image_name>

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
COPY .dvc/ .dvc/

RUN dvc remote add -d myremote gs://mlops_project_data_bucket/
RUN dvc config core.no_scm true

RUN dvc pull --verbose

COPY models/ models/

RUN pip install . --no-deps --no-cache-dir

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "80"]
