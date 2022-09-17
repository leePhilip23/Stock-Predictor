from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_stock_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_stock_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_stock_price', methods=['GET', 'POST'])
def predict_stock_price():
    firstPredictDate = str(request.form['firstDate']))
    secondPredictDate = str(request.form['location']))

    response = jsonify({
        'estimated_price': util.get_estimated_price(firstPreictDate, secondPredictDate)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Stock Price Prediction...")
    util.load_saved_data()
    app.run()
