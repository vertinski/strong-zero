![alt text](https://github.com/vertinski/strong-zero/blob/main/images/social_image01.png "strong.zero")
  
# 📎 strong.zero-0.1.9 
## Generative Multi Layer Perceptron 🤖 
<br/>

This is a simple Generative Multi Layer Perceptron (G-MLP) for time series learning and generation. Written using Numpy, and presenting many interesting concepts. 
The loss function is totally custom, and learning schedule is hard-coded for now. This will be updated.

Weights are initialized using He-et-al Initialization. 

Training process involves adding uniform noise to training data, to help the model to generalize and increase robustness. Training also includes scheduled periodic learning rate pumps, which result in noticeable Loss decreases. 
<br/>
<br/>

## The Combing and Generation process

The generation process involves "data combing". Strong.Zero "combs" the available time series data using simple scan heads -- sparse data points sampled from existing data. Then it generates the next chunk of time series points, which it adds to existing data, and repeats the process with now updated dataset. 
> Input and output data can be tokenized. 

![alt text](https://github.com/vertinski/strong-zero/blob/main/images/g-mlp_01.png "Fig. 1")
<div align="center">
    Fig. 1
</div>
<br/>

## Training process

Training schedule consists of periodic Learning Rate increases and gradual Noise decrease. A little bit of added noise is left till the end of training. The noise used for data augmentation is uniform pseudo-random noise. 

![alt text](https://github.com/vertinski/strong-zero/blob/main/images/training_01.png "Fig. 2")
<div align="center">
    Fig. 2
</div>
<br/>

✅ For now you can train the model on a CPU (test data included in code) in a couple of minutes and see the example data generated. 

🚫 The weight saving and loading needs to be repaired. 
