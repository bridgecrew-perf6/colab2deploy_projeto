"""
Date: 29 May 2022
Create API
"""
# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import joblib
import os
import wandb
import sys
from source.api.pipeline import FeatureSelector, CategoricalTransformer

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
#setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# name of the model artifact
artifact_model_name = "primeiro_projeto/model_export:latest"

# initiate the wandb project
run = wandb.init(project="primeiro_projeto",job_type="api")

# create the api
app = FastAPI()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Car(BaseModel):
    buying: str
    maint: str
    doors: str
    persons: str
    lug_boot: str
    safety: str


    class Config:
        schema_extra = {
            "example": {
                "buying": 'high',
                "maint": 'high',
                "doors": '3',
                "persons": '4',
                "lug_boot": 'med',
                "safety": 'med'
            }
        }

# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Hello World</strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
        """acquired in the Deploying a Scalable ML Pipeline in Production course to develop """\
        """a classification model on publicly available"""\
        """<a href="http://archive.ics.uci.edu/ml/datasets/Adult"> Census Bureau data</a>.</span></p>"""

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(car: Car):
    
    # Download inference artifact
    model_export_path = run.use_artifact(artifact_model_name).file()
    pipe = joblib.load(model_export_path)
    
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to
    # pass the Index.
    df = pd.DataFrame([car.dict()])

    # Predict test data
    predict = pipe.predict(df)

    return get_class(predict[0])


def get_class(value):
    if value == 0:
        return "acc"
    if value == 1:
        return "good"
    if value == 2:
        return "unacc"
    return "very good"