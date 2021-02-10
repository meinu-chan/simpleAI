import numpy as np

class AITrainer:

    def __init__(self, f_name):
        self.f_name = f_name
        self.weight = self.file_getter()
        print(self.weight)

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def Backpropagation(self, training_inputs, training_outputs, synaptic_weights):

        try:
            if(synaptic_weights == 0):
                synaptic_weights = np.random.random((len(training_inputs[0]),1))
        except:
            pass

        for i in range(20000):
            print("AI step", i)
            input_layer = training_inputs
            outputs = self._sigmoid(np.dot(input_layer, synaptic_weights))

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
            file = open(self.f_name)
        except:
            return 0
        # try:
        #     file = open(self.f_name)
        #     if(len(file.read()) == 0):
        #         raise Exception
        # except:
        #     file = open(self.f_name)

        
        try:
            data = [float(num) for num in file.read().split(",")[:-1]]
        except Exception as err:
            print(err)
            file.close()
            raise err 

        file.close()
            
        return np.array([data]).T

    def start_training(self, train_data):
        _input = np.array([_in['_input'] for _in in train_data])
        _output = np.array([[_out['_output'] for _out in train_data]]).T
        
        self.Backpropagation(_input, _output, self.weight)
        return

    def get_answer(self, _input):
        answer = self._sigmoid(np.dot(_input, self.weight))
        return answer


        