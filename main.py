from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = load_model('model.keras')

except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please ensure that 'model.pkl' exists.")

@app.post('/model_api')
def predict():

    scaler=MinMaxScaler(feature_range=(0,1))
    timeinterval=24
    prediction=12

    testapi='https://api.twelvedata.com/time_series?symbol=BTC/INR&interval=5min&outputsize=5000&apikey=e76157c75c3a42649e168c5c206e88ca'
    testdata=requests.get(testapi).json()
    testdatafinal=pd.DataFrame(testdata['values'])
    testinputs=testdatafinal['close'].values
    testinputs=testinputs.reshape(-1,1)
    modelinputs=scaler.fit_transform(testinputs)

    x_test=[]
    for x in range(timeinterval,len(modelinputs)):
        x_test.append(modelinputs[x-timeinterval:x,0])

    x_test=np.array(x_test)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    prediction_price=model.predict(x_test)
    prediction_price=scaler.inverse_transform(prediction_price)

    lastdata=modelinputs[len(modelinputs)+1-timeinterval:len(modelinputs)+1,0]
    lastdata=np.array(lastdata)
    lastdata=np.reshape(lastdata,(1,lastdata.shape[0],1))
    prediction=model.predict(lastdata)
    prediction=scaler.inverse_transform(prediction)
    return {"prediction": predict}

