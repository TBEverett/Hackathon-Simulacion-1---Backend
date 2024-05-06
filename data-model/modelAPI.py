# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import numpy as np
from lime import lime_tabular

model = load('model_dump.joblib') 

# Initialize FastAPI app
app = FastAPI()

# Define request body schema
class InputData(BaseModel):
    index: float
    # Add more features as needed

# Define prediction endpoint
@app.post("/predict/")
def predict(data: InputData):

    data = pd.read_csv('german.csv', header=None)
    data = pd.get_dummies(data=data, columns=[0,2,3,5,6,8,9,11,13,14,16,18,19])
    data.columns = data.columns.astype(str)
    data = data.drop(["20"],axis=1)

    print(data.iloc[1])
    # Make predictions
    prediction = model.predict(np.array(data.iloc[1]).reshape(1,-1))
    
    # Return predictions
    return {"predictions": prediction.tolist()}

@app.post("/explain/")
def explain(data: InputData):

    data = pd.read_csv('german.csv', header=None)
    data = pd.get_dummies(data=data, columns=[0,2,3,5,6,8,9,11,13,14,16,18,19])
    data.columns = data.columns.astype(str)
    data = data.drop(["20"],axis=1)

    explainer = lime_tabular.LimeTabularExplainer(
        data.values,
        feature_names=data.columns,
        class_names=["20"],
        verbose=True,
        mode="classification",
    )

    index = 0
    instance = data.iloc[index]
    instance_y = data.iloc[index]

    predict_fn_rf = lambda x: model.predict_proba(x).astype(float)

    local_explanation = explainer.explain_instance(instance, predict_fn_rf, num_samples=1000)
    return {"explanation": local_explanation.as_list()}


