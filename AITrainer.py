import numpy as np

class AITrainer:

    def __init__(self, f_name):
        self.f_name = f_name
        self.weight = self.file_getter()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def Backpropagation(self, training_inputs, training_outputs, synaptic_weights):

        if(synaptic_weights == 0):
            synaptic_weights = np.random.random((len(training_inputs[0]),1))

        for i in range(20000):
            print("AI step", i)
            input_layer = training_inputs
            outputs = self.sigmoid(np.dot(input_layer, synaptic_weights))

            err = training_outputs - outputs
            adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

            synaptic_weights += adjustments
        return self.file_writer(synaptic_weights.T[0])

    def file_writer(self, weights):
        file = open(self.f_name, "w")

        for weight in weights:
            file.write(f"{weight},")
        file.close()

    def file_getter(self):
        try:
            file = open(self.f_name, "r")
        except:
            return 0
        data = file.read()

        try:
            data = [float(num) for num in data.split(",")[:-1]]
        except Exception as err:
            print(err)
            return err 
            
        return np.array([data]).T

    def start_training(self, train_data):
        _input = [_in['_input'] for _in in train_data]
        _output = np.array([[_out['_output'] for _out in train_data]]).T
        
        self.Backpropagation(_input, _output, self.weight)
        return


        