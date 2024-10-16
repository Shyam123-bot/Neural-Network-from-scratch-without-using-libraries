import numpy as np
import time

class DeepNeuralNetwork():
    def __init__(self,sizes,activation='sigmoid'):
        self.sizes=sizes

        #Choose your activation function
        if activation == 'relu':
            self.activation=self.relu
        if activation == 'sigmoid':
            self.activation=self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        #Save all weights
        self.params = self.initialize()
        #Save all intermediate values, i.e. activations
        self.cache = {}
    
    def relu(self,x,derivative=False):
        '''
            Derivative of ReLU is a bit more complicated since it is not differentiable at x = 0
        
            Forward path:
            relu(x) = max(0, x)
            In other word,
            relu(x) = 0, if x < 0
                    = x, if x >= 0

            Backward path:
            ∇relu(x) = 0, if x < 0
                     = 1, if x >=0
        '''
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0,x)
    
    def sigmoid(self,x,derivative=False):
        '''
            Forward path:
            σ(x) = 1 / 1+exp(-z)
            
            Backward path:
            ∇σ(x) = exp(-z) / (1+exp(-z))^2
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def softmax(self,x):
        '''
            softmax(x) = exp(x) / ∑exp(x)
        '''
        #Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    
    def initialize(self):
        #number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]

        params={
            "W1":np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1":np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2":np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2":np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer),
        }
        return params
    
    def initialize_momemtum_optimizer(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
        }
        return momemtum_opt
    
    def feed_forward(self, x):
        '''
            y = σ(wX + b)
        '''
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]

