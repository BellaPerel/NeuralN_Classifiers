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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

batch_size = 200

train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False,
                           transform=transform,
                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



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
        #y = self.one_hot(y, 10)
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


def make_p(forward_batch):
    return forward_batch.max(axis=1)[1]


model = Neural_Network(784, 10, 200)
num_epochs = 5
batch_size = 200
train_acc = []
test_acc = []
count = 0
for epoch in range(num_epochs):
    count_train = 0
    count_test =0
    accuracy_test = 0
    accuracy_train = 0
    for i, (images, lables) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        forward_train = model.forward(images)
        model.backward(images, lables, forward_train)
        forward_train = model.forward(images)
        prediction_train = make_p(forward_train)
        accuracy_train = (prediction_train == lables).sum()+accuracy_train
        accuracy_train = np.float16(accuracy_train)
        count_train = count_train + images.size(0)
    #print(accuracy_train/count_train)
    train_acc.append(accuracy_train/count_train)

    for i, (images2, labels2) in enumerate(test_loader):
        images2 = images2.view(-1, 28*28)
        forward_test = model.forward(images2)
        prediction_test = make_p(forward_test)
        accuracy_test = (prediction_test == labels2).sum()+accuracy_test
        accuracy_test = np.float16(accuracy_test)
        count_test = count_test + images2.size(0)
    #print(accuracy_test/count_test)
    test_acc.append(accuracy_test/count_test)

# plot test_acc and train_acc vs epoch in the same plot
plt.plot(range(0,5), train_acc, label='train_acc')
plt.plot(range(0,5), test_acc, label='test_acc')
#add ticks 0,1,2,3,4,5
plt.xticks(np.arange(0, 5, 1))
plt.title('train_acc vs test_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

with open("q2_final_model.pkl", "wb") as f:
    pickle.dump(model, f)

