# Deep Learning Neural Network from Scratch

A complete implementation of a multi-layer neural network built from scratch using PyTorch for MNIST digit classification. This project demonstrates fundamental deep learning concepts including forward propagation, backpropagation, regularization techniques, and batch normalization.

## Features

- **Neural Network Architecture**: Customizable multi-layer perceptron with configurable layer dimensions
- **Activation Functions**: ReLU for hidden layers, Softmax for output layer
- **Regularization Techniques**: 
  - L2 regularization to prevent overfitting
  - Batch normalization for improved training stability
- **Training Features**:
  - Mini-batch gradient descent
  - Early stopping mechanism
  - Cross-entropy loss with regularization
- **Comprehensive Evaluation**: Training, validation, and test accuracy metrics


## Implementation Details

### Neural Network Components

1. **Forward Propagation**
   - `initialize_parameters()`: Xavier/He initialization for weights
   - `linear_forward()`: Linear transformation (Z = WA + b)
   - `relu()` and `softmax()`: Activation functions
   - `L_model_forward()`: Complete forward pass through the network

2. **Backward Propagation**
   - `Linear_backward()`: Gradients for linear layer
   - `relu_backward()` and `softmax_backward()`: Activation-specific gradients
   - `L_model_backward()`: Complete backward pass

3. **Training Process**
   - `L_layer_model()`: Main training loop with batch processing
   - `Update_parameters()`: Parameter updates using gradient descent
   - `compute_cost()`: Cross-entropy loss with L2 regularization

4. **Evaluation**
   - `Predict()`: Accuracy calculation on test data
   - `Dataset_Loader()`: MNIST data preprocessing and splitting



## Experiments

The project includes three main experiments:

1. **Baseline Model**: Standard neural network without regularization
2. **Batch Normalization**: Network with batch normalization after ReLU activations
3. **L2 Regularization**: Network with L2 weight decay regularization

## Results

The implementation provides:
- Training and validation accuracy curves
- Cost function visualization over iterations
- Comparative analysis of different regularization techniques
- Weight value comparisons between regularized and non-regularized models

## Network Architecture

**Default Configuration:**
- Input Layer: 784 neurons (28×28 MNIST images)
- Hidden Layers: 20 → 7 → 5 neurons
- Output Layer: 10 neurons (digit classes)
- Activation: ReLU (hidden), Softmax (output)

## Key Features

### Regularization Techniques
- **L2 Regularization**: Adds penalty term to prevent overfitting
- **Batch Normalization**: Normalizes layer inputs to improve training stability
- **Early Stopping**: Monitors validation accuracy to prevent overfitting

