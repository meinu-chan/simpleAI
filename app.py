from flask import Flask, render_template, request, make_response
import numpy as np
import threading
import concurrent.futures
import matplotlib.pyplot as plt

from AI import predict_digit

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route("/data", methods=['POST'])
def get_new_data():
    vector = request.get_json()['vector']

    predict = predict_digit(vector)
    print(predict)

    return predict

if __name__ == "__main__":
    app.run(debug = True)