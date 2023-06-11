
import numpy as np

class Linear:
    """Apply linear transform. Create weights and biases, 
    which shapes are weight: [in_features, out_features], biases: [out_features]
    The weights are initialized to normal distribution multiplied by 0.1 and
    biases are uniform distribution, multiplied by 0.1.
    Args:
        in_features (int): The number of features at input
        out_features (int): The number of output features (number of neurons at output)
    """
    def __init__(self, 
                 in_features:int, 
                 out_features:int):
        self.weights = 0.1 * np.random.randn(in_features, out_features)
        self.biases = 0.1 * np.random.uniform(-1, 1, out_features)
        # self.biases = np.zeros(out_features)
    
    def forward(self, inputs):
        """Calculate linear transform along input data with weights and biases
        The equation of linear transform is $y = xA + b$"""
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Calculate difference of `dvalues` by weights and biases. 
        The `dvalues` are output of before layer while back propagation
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
        

class ReLU:
    """
    Apply ReLU function.
    Forward method of ReLU object returns maximum value between 0 and input.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        return dvalues * np.where(self.output > 0, 1, 0)


class Sigmoid:
    """
    Apply Sigmoid function.
    Forward method of Sigmoid returns value between 0 and 1.
    If input much lager than 0, results will close to 1
    and lower than 0, then 0.
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, dvalues):
        return dvalues * self.output * (1 - self.output)


class LinearModel:
    """Apply linear transform. 
    Between layers, apply ReLU activation function
    And apply sigmoid function to output of linear transform.
    
    Args:
        in_features (int): The number of features at input
        hidden_dim (int): The number of hidden layer's neuron.
        out_features (int): The number of output features (number of neurons at output)
    """
    def __init__(self,
                 in_features:int,
                 hidden_dim:int,
                 out_features:int):
        self.layers = [Linear(in_features=in_features,
                              out_features=hidden_dim),
                       ReLU(),
                       Linear(in_features=hidden_dim, # 입력층과 출력층 사이의 은닉층
                              out_features=hidden_dim),
                       ReLU(),
                       Linear(in_features=hidden_dim,
                              out_features=out_features),
                       Sigmoid()]

    def forward(self, x):
        for i in range(len(self.layers)): # Linear 객체를 가져오면서 그 Linear 층을 통과시킨다.
            temp = self.layers[i] 
            x = temp.forward(x)
        return x

    def backward(self, dvalues):
        for i in range(len(self.layers)-1, -1, -1):
            temp = self.layers[i]
            dvalues = temp.backward(dvalues)
        return dvalues
