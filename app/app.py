# app/app.py

import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from mlflow.tracking import MlflowClient

# Use the value of MLFLOW_TRACKING_URI if it's set in the environment (e.g., in Docker), otherwise fall back to "file://mlruns" for local dev.
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")) #f"file://{Path('mlruns').resolve()}"))


# Load MLflow model (from registered model in "prod" stage)
#model = mlflow.pyfunc.load_model("models:/challenger_model@prod")
model = mlflow.pyfunc.load_model("model_artifacts")

app = FastAPI()

# Define input schema for a single observation
class InputData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Define output schema
class PredictionResponse(BaseModel):
    predicted_class: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: InputData):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])

    preds = model.predict(input_df)

    # Get probability if possible
    if hasattr(model._model_impl, "predict_proba"):
        proba = model._model_impl.predict_proba(input_df)
        class_index = int(np.argmax(proba[0]))
        class_proba = float(proba[0][class_index])
    else:
        class_index = int(preds[0])
        class_proba = 1.0

    return {"predicted_class": class_index, "probability": class_proba}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Run the script for testing:
    # python3 -m src.deploy.local_deployment
    # In another terminal send the first row in X_test.json to the server:
    """
    curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
       "Time": -0.5073720321, "V1": -16.5265065691, "V2": 8.5849717959, "V3": -18.6498531852,
       "V4": 9.5055935151, "V5": -13.7938185271, "V6": -2.8324042994, "V7": -16.701694296,
       "V8": 7.5173439037, "V9": -8.5070586368, "V10": -14.1101844415, "V11": 5.2992363496,
       "V12": -10.8340064815, "V13": 1.6711202533, "V14": -9.3738585836, "V15": 0.3608056416,
       "V16": -9.8992465408, "V17": -19.2362923698, "V18": -8.3985519949, "V19": 3.1017353689,
       "V20": -1.5149234353, "V21": 1.1907386948, "V22": -1.127670009, "V23": -2.3585787698,
       "V24": 0.673461329, "V25": -1.4136996746, "V26": -0.4627623614, "V27": -2.0185752488,
       "V28": -1.0428041697, "Amount": 4.7815272829
    }' | jq
    """
