import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import torch.nn as nn
import pickle


# Q 2_2
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


def load_data_q2_2():

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST('/files/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_set.__len__(), shuffle=False)

    return test_loader


def build_sets(test_loader):
    test = enumerate(test_loader)
    _, (test_images, _) = next(test)
    random_labels = bernoulli.rvs(0.5, size=len(test_images))
    random_labels_tensor = torch.Tensor(random_labels).long()

    return test_images, random_labels_tensor


def calc_acc_q2_2(images, labels, preds):
    count = images.size(0)
    correct = 0
    for label, pred in zip(labels, preds):
        if label == pred:
            correct = correct + 1
    return correct / count


def evaluate_hw1():
    test_loader = load_data_q2_2()
    test_images, random_labels_tensor = build_sets(test_loader)
    # saved model
    model = pickle.load(open("q2_2_final_model.pkl", 'rb'))
    # average acc on test
    images = test_images.view(-1, 28 * 28)
    labels = random_labels_tensor
    outputs = model.forward(images)
    _, preds = torch.max(outputs.data, 1)
    avg_acc = calc_acc_q2_2(images, labels, preds)
    #print("average accuracy is: {}%".format(round(avg_acc*100, 2)))
    return avg_acc


evaluate_hw1()
