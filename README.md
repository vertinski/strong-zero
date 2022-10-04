# 📎 strong.zero-0.1.9 
## Generative Multi Layer Perceptron 🤖 

This is a simple Generative Multi Layer Perceptron (G-MLP) for time series learning and generation. Written using Numpy, and presenting many interesting concepts. 
The loss function is totally custom, and learning schedule is hard-coded for now. This will be updated.

Weights are initialized using He-et-al Initialization. 

Training process involves adding uniform noise to training data, to help the model to generalize and increase robustness. Training also includes scheduled periodic learning rate pumps, which makes noticeable Loss decreases. 

## The Generation and Combing process

The generation process involves "data combing". Strong.Zero "combs" the available time series data using simple scan heads -- sparse data points sampled from existing data. Then it generates the next chunk of time series points, which it adds to existing data, and repeats the process with now updated dataset. 
> Input and output data can be tokenized. 

![alt text](https://github.com/vertinski/strong-zero/blob/main/images/g-mlp_01.png "Fig 1")

✅ For now you can train the model on a CPU (test data included in code) in a couple of minutes and see the example data generated. 

🚫 The weight saving and loading needs to be repaired. 
