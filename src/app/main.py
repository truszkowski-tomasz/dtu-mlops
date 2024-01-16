"""
To run the app locally, use the following command in the terminal:

uvicorn --reload --port 8000 src.app.main:app

"""

from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import time

app = FastAPI()
templates = Jinja2Templates(directory="src/app/templates/")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(request: Request, news: str = Form(...)):
    time.sleep(2)  # Simulate running process
    predicted_label = True  # There will be model prediction here

    prediction_message = "REAL!" if predicted_label else "FAKE!"

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
