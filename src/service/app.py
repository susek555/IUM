import random
from datetime import datetime

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI

from src.service.logger import CSVLogger
from src.service.model import PredictionData

csv_logger = CSVLogger(
    "./logs/service.log", ["level", "timestamp", "model", "prediction", "real"]
)

BASE_MODEL = joblib.load("./models/linear_pipeline.joblib")
ADVANCED_MODEL = joblib.load("./models/forest_pipeline.joblib")

app = FastAPI()


@app.post("/predict")
async def predict_price(data: PredictionData):
    random_number = random.uniform(0, 1)
    if random_number < 0.5:
        model = BASE_MODEL
        model_name = "base"
    else:
        model = ADVANCED_MODEL
        model_name = "advanced"

    data_dict = data.model_dump()
    prediction = model.predict(pd.DataFrame([data_dict]))
    csv_logger.log(
        level="INFO",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model=model_name,
        prediction=f"{prediction[0]:.4f}",
        real=data.price,
    )
    return {"prediction": float(prediction[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
