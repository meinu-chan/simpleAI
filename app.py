from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt

from AI import predict_digit

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route("/data", methods=['POST'])
def get_new_data():
    vector = request.get_json()['vector']

    # plt.figure()
    # plt.imshow(vector)
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    predict_digit(vector)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug = True)
