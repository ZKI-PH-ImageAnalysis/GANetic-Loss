# GANetic-Loss
This repository contains the code for the new loss function found by using the Genetic Programming (GP) approach. The loss function can be used to train GAN models.

The full paper can be downloded [here]().

![fig_func_plot-1](https://github.com/ZKI-PH-ImageAnalysis/GANetic-Loss/assets/107623498/6e0a7299-b904-46be-9dae-a766e8f949ff)

# Example
The repository contains an example of how GANetic loss can be used to train the small DCGAN model on CIFAR-10 dataset. 

# Using GANetic Loss in your own project
The repository contains two implementations of the GANetic loss: in pytorch and in tensorflow. Both versions can be used to train deep learning models for image classification. 

# Results
GANetic loss was used to train the small DCGAN model on CIFAR-10 dataset. DCGAN was trained with the batch size of 128 for 200 epochs. The results of 25 independent program runs were compared to the identical model using binary cross entropy, hinge, adversarial, Wasserstein, and least square loss functions. 

Corresponding FID values (the best, the worst, the mean) and their standard deviation: 

| Loss | Adversarial | BCE | Hinge | Wasserstein | LeastSquare | GANetic | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Best | 9.45 | 2.06 | 7.11 | 3.04 | 4.60 | **1.52** |
| Worst | 72.58 | 45.30 | 62.47 | 31.74 | 80.87 | **2.76** |
| Mean | 18.52 | 8.34 | 12.41 | 12.93 | 17.52 | **2.20** |
| STD | 16.12 | 8.92 | 10.59 | 4.85 | 15.81 | **0.39** |

The variation of FID values for the tested loss functions over 25 runs:

![fig_mode_col-1](https://github.com/ZKI-PH-ImageAnalysis/GANetic-Loss/assets/107623498/55c395a6-bb20-4118-b36b-bf46b6954e86)

Examples of the generated images:

![image](https://github.com/ZKI-PH-ImageAnalysis/GANetic-Loss/assets/107623498/479f7df0-b1d5-4240-98e6-ea4aff8771a7)

See paper for details.
