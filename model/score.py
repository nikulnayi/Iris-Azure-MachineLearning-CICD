# model/score.py

import joblib
import json
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        inputs = np.array(data["data"])
        predictions = model.predict(inputs).tolist()
        return {"result": predictions}
    except Exception as e:
        return {"error": str(e)}
