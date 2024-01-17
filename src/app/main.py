"""
To run the app locally, use the following command in the terminal:

uvicorn --reload --port 8000 src.app.main:app

"""

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from src.predict_model import predict
import pandas as pd
import hydra
import time

app = FastAPI()
templates = Jinja2Templates(directory="src/app/templates/")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(request: Request, news: str = Form(...)):

    with hydra.initialize(config_path="../config", version_base="1.1"):
        cfg = hydra.compose(config_name="default_config.yaml", overrides=[f"predict.texts=['{news}']"])

        predictions = predict(cfg)

        print(predictions)
        print(type(predictions))

        prediction_message = "REAL!" if predictions else "FAKE!"

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "input_news": news,
                "prediction": f"Given news is likely to be... {prediction_message}",
            },
        )


@app.get("/predict/", response_class=RedirectResponse)
async def redirect_to_home():
    return RedirectResponse(url="/")
