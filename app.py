from flask import Flask, render_template, request
import AITrainer
import numpy as np

app = Flask(__name__)
AI = AITrainer.AITrainer("weight")
data = []

@app.route('/')
def hello():
    return render_template("index.html")

@app.route("/data", methods=['POST'])
def get_new_data():
    global data

    _input = request.args['input'].split(",")
    _output = request.args['output']
    data.append({"_input": [int(num) for num in _input], "_output": int(_output)})
    # AI.Backpropagation(training_inputs = data['_input'], training_outputs = data['_output'])
    return render_template("index.html")

@app.route("/answer", methods=['GET'])
def show_answer():
    _vector = request.args['vector'].split(",")
    data = np.array([[int(num) for num in _vector]])
    print(AI.get_answer(data))
    return render_template("index.html")

@app.route("/train", methods=['GET'])
def train():
    global data
    if(len(data) >= 2):
        AI.start_training(data)
    return render_template("index.html")

@app.after_request
def after_request_func(response):
    return response


if __name__ == '__main__':
    app.run(debug = True)