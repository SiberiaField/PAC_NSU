import numpy as np

class DigitClassificator:
    def __init__(self, data, biases):
        self.weights = []
        for digit in range(10):
            avg_digit = self.average_digit(data, digit)
            digit_weights = np.transpose(avg_digit)
            self.weights.append(digit_weights)
        
        self.biases = np.zeros(10)
        for i in range(len(biases)):
            if i == 10:
                break
            self.biases[i] = biases[i]

    def average_digit(self, data, digit):
        filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
        filtered_array = np.asarray(filtered_data)
        return np.average(filtered_array, axis=0)

    def binary_step(self, x):
        return 1 if x >= 0 else 0
    
    def set_biases(self, index, values):
        if len(index) != len(values):
            raise ValueError('index size is not equal to values size')
        self.biases[index] = values

    def predict_by_digit(self, digit, x):
        answer = np.dot(self.weights[digit], x)[0][0]
        answer /= np.linalg.norm(self.weights[digit])
        answer += self.biases[digit]
        return answer
    
    def predict(self, x):
        raw_predict = np.zeros((10, 1))
        for digit in range(10):
            raw_predict[digit] = self.predict_by_digit(digit, x)
        return raw_predict

