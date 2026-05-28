# 🧠 Neural Network Engine from Scratch (NumPy)

## Overview
This repository contains a fully functional Deep Learning engine built entirely from scratch using only Python and **NumPy**. 

Instead of relying on modern frameworks like PyTorch or TensorFlow, this project implements the core mathematics of Artificial Intelligence—including Forward Propagation, Log-Loss calculation, and Backpropagation (via the Calculus Chain Rule)—to recognize handwritten digits from the MNIST dataset.

## 🚀 The Learning Journey (Project Structure)

This project is divided into two phases to demonstrate the evolution from a basic linear model to a true Deep Neural Network.

### Phase 1: Single-Layer Perceptron (`01_Single_Layer_Network.ipynb`)
A foundational Multi-Class linear model that attempts to recognize digits using only the raw pixel data.
* **Architecture:** 784 Inputs $\rightarrow$ 10 Outputs
* **Math:** Matrix Multiplication ($Wx + b$) and Softmax activation.
* **Optimization:** Mini-Batch Gradient Descent.
* **Accuracy:** ~89% 
* **Limitation:** Hit a "glass ceiling" because a flat layer can only read raw pixels, not geometric shapes.

### Phase 2: Deep Neural Network (`02_Deep_Neural_Network.ipynb`)
To break the 89% accuracy ceiling, a Hidden Layer was introduced, allowing the network to detect non-linear patterns (curves, edges, loops).
* **Architecture:** 784 Inputs $\rightarrow$ 128 Hidden Neurons $\rightarrow$ 10 Outputs
* **Math:** Implemented the **ReLU** activation function and derived the **Calculus Chain Rule** by hand for the backward pass to update weights across multiple layers.
* **Accuracy:** ~97%

## ⚙️ How to Run
Both models were built in Google Colab for easy execution.
1. Clone this repository or download the `.ipynb` files.
2. Upload the notebooks to [Google Colab](https://colab.research.google.com/).
3. Run the cells sequentially. The code automatically downloads the Keras MNIST dataset for the raw pixel matrices.

## 💡 Key Takeaways
Building this engine without high-level frameworks solidified my understanding of:
- **Data Normalization:** Flattening 28x28 images into matrices and scaling pixel values.
- **Gradient Descent:** How the learning rate and batch sizes impact training stability and speed.
- **Backpropagation:** Passing the "blame" (error) backward through a network using matrix transposition and derivatives.

## ⏭️ Next Steps
The next phase of my AI journey will be upgrading from raw NumPy to **PyTorch** to build **Convolutional Neural Networks (CNNs)**. This will allow the model to recognize digits in real-world, messy photographs by using spatial filters instead of flattening the image.
