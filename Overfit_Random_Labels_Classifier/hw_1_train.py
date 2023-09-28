import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import torch.nn as nn
from torch.autograd import Variable
import pickle


# Question 2_2
class Neural_Network_2_2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Neural_Network_2_2, self).__init__()
        self.func1 = nn.Linear(input_size, 200)
        self.relu = nn.ReLU()
        self.func2 = nn.Linear(200, num_classes)

    def forward(self, x):
        out = self.func1(x)
        out = self.relu(out)
        out = self.func2(out)
        return out


def build_sets(train_loader, test_loader):
    # extracting first 128 images and generating labels
    train = enumerate(train_loader)
    _, (train_images, labels) = next(train)
    random_labels_train = bernoulli.rvs(0.5, size=128)
    random_labels_train_tensor = torch.Tensor(random_labels_train).long()

    test = enumerate(test_loader)
    _, (test_images, _) = next(test)
    random_labels_test = bernoulli.rvs(0.5, size=len(test_images))
    random_labels_test_tensor = torch.Tensor(random_labels_test).long()

    return train_images, random_labels_train_tensor, test_images, random_labels_test_tensor



def load_data_q2_2():
    batch_size = 128

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('/files/',train = True,transform=transform,download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size, shuffle=False)

    test_set = datasets.MNIST('/files/',train = False,transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size= test_set.__len__(),shuffle=False)

    return train_loader, test_loader


def calc_accuracy_q2_2(images, labels, preds, accuracy_list):
    count = images.size(0)
    correct = 0
    for label, pred in zip(labels, preds):
        if pred == label:
            correct = correct + 1
    accuracy_list.append(correct / count)


def plot_loss(epochs_list, train_loss_list, test_loss_list):
    plt.plot(epochs_list, train_loss_list, label = "train loss", color = "pink")
    plt.plot(epochs_list, test_loss_list, label = "test loss", color = "blue")
    plt.legend()
    plt.title("Loss of train and test Q2_2", fontweight='bold')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def learn_predict_q2_2(num_epochs):
    loss_test = []
    loss_train = []
    accuracy_train, accuracy_test = [],[]
    train_loader, test_loader = load_data_q2_2()
    train_images, random_labels_train_tensor, test_images, random_labels_test_tensor = build_sets(train_loader, test_loader)
    model = Neural_Network_2_2(28*28, 2)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        images = train_images.view(-1, 28 * 28)
        labels = random_labels_train_tensor
        # Train model
        optimizer.zero_grad()
        outputs = model.forward(images)
        _, preds = torch.max(outputs.data, 1)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        calc_accuracy_q2_2(images, labels, preds, accuracy_train)
        # Test Model
        images = test_images.view(-1, 28 * 28)
        labels = random_labels_test_tensor
        outputs = model.forward(images)
        _, preds = torch.max(outputs.data, 1)
        loss = loss_func(outputs, labels)
        calc_accuracy_q2_2(images, labels, preds, accuracy_test)
        loss_test.append(loss.item())

    with open("q2_2_final_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return [i for i in range(1, num_epochs + 1)], loss_train, loss_test, accuracy_test, accuracy_train


def plot_accuracy_q2_2(epochs_list, train_accuracy_list, test_accuracy_list):
    plt.plot(epochs_list, train_accuracy_list, label="train accuracy", color="pink")
    plt.plot(epochs_list, test_accuracy_list, label="test accuracy", color="blue")
    plt.title("Accuracy of train and test Q2_2 ", fontweight='bold')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()


def q2_2():
    epochs, train_loss, test_loss, test_accuracy, train_accuracy = learn_predict_q2_2(900)
    plot_loss(epochs, train_loss, test_loss)
    plot_accuracy_q2_2(epochs, train_accuracy, test_accuracy)
    #print mean test loss
    #print(f"mean test loss value of Q2_2 is: {sum(test_loss) / len(train_loss)}")


q2_2()

