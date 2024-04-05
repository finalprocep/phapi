from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import initializers

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
    bit_model = load_model('model.keras')
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please ensure that 'model.keras' exists.")

@app.post('/model_api')
def predict():
    try:
        now = datetime.now()
        now.replace(microsecond=0)

        def get_exchange_data():
            exchangeapi = 'https://api.twelvedata.com/exchange_rate?symbol=BTC/INR&timezone=Asia/Kolkata&apikey=e76157c75c3a42649e168c5c206e88ca'
            exchangedata = requests.get(exchangeapi)
            if exchangedata.status_code != 200:
                raise HTTPException(status_code=exchangedata.status_code, detail="Failed to fetch exchange data")
            realtimeprice = exchangedata.json().get('rate')
            return realtimeprice

        enddate = (now - timedelta(days=0)).replace(microsecond=0)
        startdate = (now - timedelta(hours=48)).replace(microsecond=0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        timeinterval = 24
        prediction = 12

        testapi = f"https://api.twelvedata.com/time_series?apikey=e76157c75c3a42649e168c5c206e88ca&interval=5min&outputsize=576&order=asc&start_date={startdate}&end_date={enddate}&format=JSON&symbol=BTC/INR&timezone=Asia/Kolkata"
        testdata = requests.get(testapi)
        if testdata.status_code != 200:
            raise HTTPException(status_code=testdata.status_code, detail="Failed to fetch test data")
        testdatafinal = pd.DataFrame(testdata.json().get('values'))
        bitcoinprice = pd.to_numeric(testdatafinal['close'], errors='coerce').values
        testinputs = testdatafinal['close'].values
        testinputs = testinputs.reshape(-1, 1)
        modelinputs = scaler.fit_transform(testinputs)

        x_test = np.array(modelinputs)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        prediction_price = bit_model.predict(x_test)
        prediction_price = scaler.inverse_transform(prediction_price)

        lastdata = modelinputs[len(modelinputs) + 1 - timeinterval:len(modelinputs) + 1, 0]
        lastdata = np.array(lastdata)
        lastdata = np.reshape(lastdata, (1, lastdata.shape[0], 1))
        prediction = bit_model.predict(lastdata)
        prediction = scaler.inverse_transform(prediction)
        predict = prediction[0][0]
        exchange = get_exchange_data()

        return {"prediction": predict, "realtime": exchange}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
