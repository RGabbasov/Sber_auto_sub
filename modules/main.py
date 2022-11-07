import dill
import os

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

project_path = 'C:/Users/GP62/PycharmProjects/pythonProject/Skillbox_diploma_proj'
pickle_file_name = 'auto_sub_prediction.pkl'
file_path = os.path.join(project_path, 'data/models/', pickle_file_name)

with open(file_path, 'rb') as file:
    model = dill.load(file)
    print(model['metadata'])


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser:  str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    is_subscribed: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    weights = model['metadata']['class_weights']
    y = (model['model'].predict_proba(df)[:, 1]*weights[1.0] >= 0.5)
    return {
        'session_id': form.session_id,
        'is_subscribed': y
    }


