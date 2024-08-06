import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=64, shuffle=False)


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=20)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
        
net = MyNetwork()


torch.nn.linear()

# 1. Create a neural network:

## a. Initialize 3 layers

## b. Define the forward function:

### i. Reshape the data to a fully connected layer. Hint: Use .view() or .flatten().

### ii. Let the input pass through the different layers.

### iii. Consider what activation function you want to use in between the layers, and for the final layer.


## c. Loss function and optimizer:
### i. Consider what loss function and optimizer you want to use.


## d. Create the training loop:

## e. Create the evaluation loop:

## f. Save the model
torch.save(MyModel.state_dict(), PATH)

# 2. Report your accuracy, is this satisfactory? Why / why not?
MyModel = MyModel()
MyModel.load_state_dict(torch.load(PATH))
MyModel.eval()

# 3. Plot the loss curve.
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()