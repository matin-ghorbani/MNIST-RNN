# MNIST-RNN

This repository contains PyTorch code for training and evaluating Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTM) networks on the MNIST dataset.

## Overview

The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). This project explores the performance of different types of recurrent neural networks on this dataset.

### Results

- **Simple RNN:**
  - Training Accuracy: ***96.34%*** (57805 / 60000)
  - Testing Accuracy: ***96.13%*** (9613 / 10000)
  
- **GRU:**
  - Training Accuracy: ***98.74%*** (59244 / 60000)
  - Testing Accuracy: ***98.24%*** (9824 / 10000)

- **LSTM:**
  - Training Accuracy: ***98.62%*** (59172 / 60000)
  - Testing Accuracy: ***98.31%*** (9831 / 10000)

## Usage

First clone the repository

```bash
git clone https://github.com/matin-ghorbani/MNIST-RNN.git
cd MNIST-RNN
```

Then install dependencies

```bash
pip install -r requirements.txt
```

Finally you can test each of them using these commands:

- **Simple RNN:**

    ```bash
    python simple_rnn.py
    ```
  
- **GRU:**

    ```bash
    python GRU.py
    ```

- **LSTM:**

    ```bash
    python LSTM.py
    ```

## Model Definitions

- **RNN**: A simple recurrent neural network.
- **GRU**: A gated recurrent unit network.
- **LSTM**: A long short-term memory network.

