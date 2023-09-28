import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math
from keras.datasets import mnist
from mlxtend.data import mnist_data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import math
import numpy as np
import pickle
import hw_1_324620814_train


def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def sigmoidPrime(s):
    # derivative of sigmoid
    # s: sigmoid output
    return s * (1 - s)


def RelU(x):
    x[x < 0] = 0
    return x


def tanh(t):
    t = np.double(t)
    return  torch.div(torch.exp(2*torch.tensor(t)) -1, torch.exp(2*torch.tensor(t)) +1)


def ReLU(Z):
    return np.maximum(Z, 0)


def ReluPrime(x):
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def tanhPrime(t):
    # derivative of tanh
    # t: tanh output
    return 1 - t*t


def softmax_mine(vec):
    vec = np.exp(vec) - max(vec)
    sum_exp = sum(vec)
    return vec/sum_exp


def softmax_mine2(Z):
    return np.apply_along_axis(softmax_mine, 1, Z)


def softmax_final(x):
    exponents = torch.exp(x - x.max(axis=1)[0].reshape((-1, 1)))
    return exponents / torch.sum(exponents, axis=1).reshape((-1, 1))

class Neural_Network:
    def __init__(self, input_size=784, output_size=10, hidden_size=200):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        #self.W1= self.W1/ torch.norm(self.W1)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = RelU(self.z1)
        #tanh_func = nn.Tanh()
        #self.h = tanh_func(self.z1)
        #self.h = tanh(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax_final(self.z2)


    def one_hot(self, Y, size):
        one_hot_Y = torch.zeros((Y.size(0), size))
        for i, val in enumerate(Y):
            one_hot_Y[i][int(val)] = 1
        return one_hot_Y

    def backward(self, X, y, y_hat, lr=.1):
        batch_size = y.size(0)
        # backward pass: compute gradients of loss according to multiclass cross entropy
        dl_dz2 = (1/batch_size)*(y_hat - self.one_hot(y,10))
        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        #dl_dz1 = dl_dh * tanhPrime(self.h)
        dl_dz1 = dl_dh * ReluPrime(self.h)
        self.W1 -= lr*torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr*torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr*torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr*torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))
        #print(self.W1, self.b1, self.W2, self.b2)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

batch_size = 200

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])


def load_data_q2_2():
    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transform,
                               download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return test_loader


def make_p(forward_batch):
    return forward_batch.max(axis=1)[1]


def evaluate_hw1():
    test_loader = load_data_q2_2()
    model = pickle.load(open("q2_final_model.pkl", 'rb'))
    num_epochs = 5
    test_acc = []
    count_test = 0
    accuracy_test = 0
    for epoch in range(num_epochs):
        count_test = 0
        accuracy_test = 0
        for i, (images, lables) in enumerate(test_loader):
            images = images.view(-1, 28*28)
            forward_test = model.forward(images)
            model.backward(images, lables, forward_test)
            prediction_test = make_p(forward_test)
            accuracy_test = (prediction_test == lables).sum()+accuracy_test
            accuracy_test = np.float16(accuracy_test)
            count_test = count_test + images.size(0)
        #print(accuracy_test/count_test)
        test_acc.append(accuracy_test/count_test)
    #return average accuracy
    return np.mean(test_acc)


evaluate_hw1()
