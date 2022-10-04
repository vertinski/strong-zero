

'''
MIT License

Copyright (c) 2022 vertinski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import numpy as np
from numpy import random
import random
import matplotlib.pyplot as plt
import math
from datetime import datetime
import pickle





class MLP:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.weights1 = np.random.randn (self.input_layer_size, self.hidden_layer_size) * np.sqrt (2 / self.input_layer_size)
        self.weights2 = np.random.randn (self.hidden_layer_size, self.hidden_layer_size) * np.sqrt (2 / self.hidden_layer_size)
        self.weights3 = np.random.randn (self.hidden_layer_size, self.output_layer_size) * np.sqrt (2 / self.hidden_layer_size)

        self.bias1 = np.zeros ((1, self.hidden_layer_size))
        self.bias2 = np.zeros ((1, self.hidden_layer_size))
        self.bias3 = np.zeros ((1, self.output_layer_size))


    def relu (self, x):
        return np.maximum (0.1 * x, x)   # Leaky ReLu


    def reluDerivative (self, x):
        x[x <= 0] = 0.1         # Sneaky relu
        return x


    def forward (self, input_layer):
    
        self.hidden_layer1 = self.relu (np.dot(input_layer, self.weights1) + self.bias1)
        self.hidden_layer2 = self.relu (np.dot(self.hidden_layer1, self.weights2) + self.bias2)
        self.output_layer = self.relu (np.dot(self.hidden_layer2, self.weights3) + self.bias3)
        return self.output_layer


    def train (self, input_layer, output_layer, learning_rate, iteration):
        # Forward
        self.hidden_layer1 = self.relu (np.dot(input_layer, self.weights1) + self.bias1)
        self.hidden_layer2 = self.relu (np.dot(self.hidden_layer1, self.weights2) + self.bias2)
        self.output_layer = self.relu (np.dot(self.hidden_layer2, self.weights3) + self.bias3)

        # Backward
        self.error3 = output_layer - self.output_layer 
        self.delta3 = self.error3 * self.reluDerivative (self.output_layer) * learning_rate
        self.error2 = np.dot (self.delta3, self.weights3.T)
        self.delta2 = self.error2 * self.reluDerivative (self.hidden_layer2) * learning_rate
        self.error1 = np.dot (self.delta2, self.weights2.T)
        self.delta1 = self.error1 * self.reluDerivative (self.hidden_layer1) * learning_rate

        # Update weights
        self.weights1 += np.dot (input_layer.T, self.delta1)
        self.weights2 += np.dot (self.hidden_layer1.T, self.delta2)
        self.weights3 += np.dot (self.hidden_layer2.T, self.delta3)
        
        # Update biases
        self.bias1 += np.sum (self.delta1, axis=0)
        self.bias2 += np.sum (self.delta2, axis=0)
        self.bias3 += np.sum (self.delta3, axis=0)


    def printWeights (self):
        print ("Weights 1:", self.weights1)
        print ("Weights 2:", self.weights2)
        print ("Weights 3:", self.weights3)


    def printBiases (self):
        print ("Biases 1:", self.bias1)
        print ("Biases 2:", self.bias2)
        print ("Biases 3:", self.bias3)


    def printOutput (self):
        print ("Output layer:", self.output_layer)


    def getOutput (self):
        return self.output_layer


    def getWeights (self):
        return self.weights1.tolist(), self.weights2.tolist(), self.weights3.tolist()


    def setWeights (self, weights):
        print ('Number of layer weights: ', len(weights))
        print()
        if len (weights) < 3:
            print ('Error loading weights......')
            print ('Number of layer weights: ', len(weights))
        self.weights1 = np.array (weights[0])
        self.weights2 = np.array (weights[1])
        self.weights3 = np.array (weights[2])


    def generate (self, input_data, iter_count):
        result = input_data.copy()  # add starting data to result list
        
        heads = [0, 1, 2, 5, 9, 11, 13, 17, 19, 23, 26, 29,
             31, 37, 41, 43, 47, 53, 61, 67, 76, 83, 95,
             111, 123, 145, 146, 147, 148, 149, 150, 151]
        combed = []

        for g in range (iter_count):
            for head in heads:
                combed.append (result[-1 - head])
            
            resultz = self.forward (np.array([combed]))[0]
            result += resultz.tolist()
            combed = []

        plt.plot (result)
        plt.show()



def train_data_gen():
    input = 0
    data = []
    data2 = []

    # Generate complex sinusoidal data
    for _ in range (0, 4000): 
        data.append ((math.sin(input) + (math.sin(3*input)/2))) # * random.uniform(0.8, 1))
        input += 0.04


    # normalize the data to [0, 1]
    data = np.array (data)
    data = data + abs (min(data))
    data = data / max (data)
    
    ### plot the dataset
    plt.plot (data, label = 'Training Data')
    plt.legend (loc = "lower right")
    plt.show()
    
    training_data = []
    result_data = []

    training_eval = []
    result_eval = []

    # prime number based "attention" heads
    heads = [0, 1, 2, 5, 9, 11, 13, 17, 19, 23, 26, 29,
             31, 37, 41, 43, 47, 53, 61, 67, 76, 83, 95,
             111, 123, 145, 146, 147, 148, 149, 150, 151]
    combed = []

    for i in range (155, len(data) - 72):
        for head in heads:
            combed.append (data[i - head])
        
        if i > len (data) - 500:  #- 500:      #i % 5 == 0:
            training_eval.append ([combed])
            result_eval.append ([data[i+1:i+33].tolist()])
        else:
            training_data.append ([combed])
            result_data.append ([data[i+1:i+33].tolist()])
        combed = []
    
    training_data = np.array (training_data)
    result_data = np.array (result_data)
    training_eval = np.array (training_eval)
    result_eval = np.array (result_eval)
    
    return training_data, result_data, training_eval, result_eval


def main():
    mlp = MLP (32, 96, 32) #parameters are: input | two-hidden-layers | output

    
    ### Menu: load the weights, etc..... ###

    print()
    choice = input ('Load weights....... [y/n] ')
    genflag = False    #this flag determines data generation (for testing the NN)

    if choice == 'y':
        with open (input('Input file name: '), 'rb') as f:
            weightsX = pickle.load (f)
            mlp.setWeights (weightsX)
            
        print()
        print ('Weights loaded...... [press Enter]')
        input ('')
        genflag = True   # if weights loaded then use data generation method
        
    elif choice == 'n':
        print ('Starting the training......')
    
    else: pass
    print()


    ### Training part ###

    input_layer, output_layer, input_eval, output_eval = train_data_gen()    # generates training data


    original_data = []
    for i in range (0, 164):
        original_data.append ((input_layer[-170 + i])[0][0])


    # Generate a dataset after loading trained weights
    if genflag == True:
        mlp.generate (original_data, 100)
    
    if genflag == True:     #exit if weights has been loaded and generated data plotted
        print()
        print ('Exiting........')
        print()
        return 0
    
    
    data_range = [int(i) for i in range(len(input_layer))]   # create element index for shuffling
    eval_range = [int(i) for i in range(len(input_eval))]


    loss = []
    train_eval = []
    epochs = 200
    
    learning_rate = 0.001
    
    noise_range = 0.4   #add random noise to each training input vector 
    noise = 0
    
    iteration = 0
    k = 0
    
    now = datetime.now()    # record training start time
    start_time = now.strftime ("%d/%m/%Y %H:%M:%S")
    
    random.seed()   # reset random seed
    
    for epo in range (epochs):
        random.shuffle (data_range)    # randomize training data index
        random.shuffle (eval_range)    # randomize evaluation data index
        eval_index = eval_range  #eval_index is shorter than data_range index -- it needs to be replenished

        for i in data_range:
            noise = np.array ([random.uniform (-noise_range, noise_range) for _ in range (32)])        
            mlp.train (input_layer[i] + noise, output_layer[i], learning_rate, iteration)   # train using SGD
            
            loss.append (np.sum(np.absolute(output_layer[i] - mlp.forward(input_layer[i]))))
            
            if len (eval_index) == 0:   #if evaluation data index is all spent (popped)
                eval_index = eval_range
                random.shuffle (eval_index)
            else: k = eval_index.pop()
            
            train_eval.append (np.sum(np.absolute(output_eval[k] - mlp.forward(input_eval[k]))))
            
            if iteration % 100 == 0:
                print ('Epoch:', epo, 'Iter:', iteration, 'LRate:', round(learning_rate, 5), 'Noise:', round(noise_range+0.0000001, 7), 'Loss:', round(loss[-1], 5))
            
            iteration += 1
            
            learning_rate *= 1.000022 
            if learning_rate > 0.01: learning_rate = 0.00004 
            noise_range -= 0.0000008
            if noise_range <= 0.002: noise_range = 0.002


    # Print out weights
    print ("Weights 1:", mlp.weights1)
    print ("Weights 2:", mlp.weights2)
    print ("Weights 3:", mlp.weights3)

    # Print out biases
    print ("Biases 1:", mlp.bias1)
    print ("Biases 2:", mlp.bias2)
    print ("Biases 3:", mlp.bias3)


    now = datetime.now()    # record training end time
    end_time = now.strftime ("%d/%m/%Y %H:%M:%S")

    print()
    print ('Start time:', start_time)
    print ('End time:  ', end_time)
    print()


    # Plot training and evaluation loss
    plt.plot (loss, label = 'Loss')
    plt.plot (train_eval, label = 'Eval')
    plt.legend (loc = "upper right")
    plt.show()


    # Generate time series data prediction
    mlp.generate (original_data, 100)



    ### Menu: save the weights, etc..... ###
    
    print()
    choice = input ('Save weights....... [y/n] ')

    if choice == 'y':
        weightsX = [1, 2, 3]
        weightsX[0], weightsX[1], weightsX[2] = mlp.getWeights()
        
        with open ('weights01.bin', 'wb') as f:
            pickle.dump (weightsX, f)
        print()
        print ('Weights saved.......')
        
    elif choice == 'n':
        print ('Exiting......')
    
    else: pass
    
    print()



if __name__ == "__main__":
    main()






