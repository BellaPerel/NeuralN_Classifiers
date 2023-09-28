**Task 1: MNIST Classifier**

  We built a neural network classifier for the MNIST dataset. In this task, we adhered to specific constraints, such as using only torch tensor manipulations and avoiding built-in functions like backward(), built-in loss functions, activations, optimization, and layers from torch.nn. Our neural network included at least one hidden layer. Our aim was to achieve a minimum accuracy of 75% on the test set.

**Task 2: Overfitting to Random Labels**

  We used the MNIST dataset with certain settings: We worked with the first 128 samples from the training dataset. We set shuffle to False. We utilized a batch size of 128. We generated random labels based on a Bernoulli distribution with a probability of ½, assigning random labels of 0 or 1 to each sample. We employed the same network architecture as in Task 1 along with cross-entropy loss to showcase overfitting. Our goal was to achieve a low loss value, close to 0, in this specific scenario. We created plots illustrating the convergence of loss for both the training and test data over epochs. We calculated the mean loss value of the test data and provided an explanation for our findings.

In these tasks, we worked on building and training neural networks, demonstrating overfitting, and provided code and documentation for reproducibility and evaluation.
