
import pickle
import json
import numpy as np

dataColumns = None
model = None

# Make predictions of certain stock
def get_estimated_price(priceWindow, X):
    q_90 = int(len(priceWindow[priceWindow['Target Dates'] > '2017-09-08']))

    x_test = X[q_90:]
    
    return round(model.predict(x_test).flatten())

# Save the data for further use  
def load_saved_data():
    print("loading saved data...start")
    global dataColumns
    global stockType

    with open("./data/columns.json", "r") as f:
        dataColumns = json.load(f)['data_columns']

    global model
    if model is None:
        with open('./data/stock_prices_model.pickle', 'rb') as f:
            model = pickle.load(f)
    print("loading saved data...done")


def get_data_columns():
    return dataColumns

if __name__ == '__main__':
    load_saved_artifacts()
