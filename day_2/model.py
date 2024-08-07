import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

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
model = MyNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

## d. Create the training loop:
# Training loop
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    model.train()
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

## e. Create the evaluation loop:
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

## f. Save the model
PATH = './mymodel.pth'
torch.save(model.state_dict(), PATH)

# 2. Report your accuracy, is this satisfactory? Why / why not?
print(f'Accuracy of the loaded model on the 10000 test images: {accuracy:.2f}%')

# 3. Plot the loss curve.
plt.figure(figsize=(10,5))
plt.plot(losses)
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()