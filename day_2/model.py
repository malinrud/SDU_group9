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


# 1. Create a neural network:

## a. Initialize 3 layers
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.relu = nn.ReLU()

## b. Define the forward function:
    def forward(self, x):
### i. Reshape the data to a fully connected layer. Hint: Use .view() or .flatten().
        x = x.view(x.size(0), -1)

### ii. Let the input pass through the different layers.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

### iii. Consider what activation function you want to use in between the layers, and for the final layer.
        x = self.fc3(x) 
        return x

## c. Loss function and optimizer:
### i. Consider what loss function and optimizer you want to use.

# Initialize the network
model = MyNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

## d. Create the training loop:
## e. Create the evaluation loop:
num_epochs = 5
loss_values = []
val_loss_values = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_training_loss = running_loss / len(train_loader)
    loss_values.append(avg_training_loss)
    
    # Validation loss
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item()
    
    avg_val_loss = val_running_loss / len(test_loader)
    val_loss_values.append(avg_val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

## f. Save the model
PATH = './mymodel.pth'
torch.save(model.state_dict(), PATH)

# Load the model
model = MyNetwork()
model.load_state_dict(torch.load(PATH))

# Evaluate the loaded model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.detach(), dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total

# 2. Report your accuracy, is this satisfactory? Why / why not?
print(f'Accuracy of the loaded model on the 10000 test images: {accuracy:.2f}%')

# 3. Plot the loss curve.
plt.plot(num_epochs, loss_values, 'bo', label='Training loss')
plt.plot(num_epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()