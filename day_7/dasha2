import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from torch.utils.data import Dataset, DataLoader
import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import IMDB
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# -------------------------------------------------Dataset---------------------------------------------------------
# Tokenize function (takes a string of text, converts it to lowercase, and splits it into individual words (tokens))
def tokenize(text):
    return text.lower().split()


# Load IMDB dataset and count tokens
train_iter, test_iter = IMDB()
counter = Counter()
for label, text in itertools.chain(train_iter, test_iter):
    tokenized_text = tokenize(text)
    counter.update(tokenized_text)

# Create vocabulary
vocab = build_vocab_from_iterator([counter.keys()], specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Define vocabulary size
vocab_size = len(vocab)
print(f'Vocabulary size: {vocab_size}')


# Define Dataset class (used to create a custom dataset)
class IMDBDataset(Dataset):
    def __init__(self, data_iter, vocab):
        self.data = list(data_iter)
        self.vocab = vocab
        self.tokenizer = get_tokenizer('basic_english')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        text_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(text)], dtype=torch.long)
        label_tensor = torch.tensor(1 if label == 'pos' else 0,
                                    dtype=torch.long)  # Convert label to 1 (positive) or 0 (negative)
        return text_tensor, label_tensor


# Define collate function (processes batches of data, combining them into a format suitable for the model)
def collate_batch(batch):
    texts, labels = zip(*batch)
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab['<unk>'])
    labels = torch.stack(labels)
    return texts_padded, labels


# Create DataLoaders
train_dataset = IMDBDataset(train_iter, vocab)
test_dataset = IMDBDataset(test_iter, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)


#-------------------------------------------------Network----------------------------------------------
# Define the model
class ThreeLayerBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=3, dropout_prob=0.5):
        super(ThreeLayerBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h_out, _ = self.lstm(x)
        h_out = h_out[:, -1, :]
        out = self.fc(h_out)
        return out


#Model Initialization
embed_size = 100
hidden_size = 32
output_size = 2

model = ThreeLayerBiLSTM(vocab_size, embed_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)


# Define loss Function with L1 L2 Regularization
def l1_l2_loss(output, target, model, l1_lambda=0.001, l2_lambda=0.001):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)

    # Add L2 regularization (weight decay)
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.sum(param ** 2)
    loss += l2_lambda * l2_reg

    # Add L1 regularization
    l1_reg = 0
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    loss += l1_lambda * l1_reg

    return loss


# Define optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)


#--------------------------------------------Training----------------------------------------
# simple Training loop
def train(model, train_loader, optimizer, l1_lambda=0.001, l2_lambda=0.001, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = l1_l2_loss(outputs, labels, model, l1_lambda, l2_lambda)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')


# Example usage
#train(model, train_loader, optimizer, num_epochs=5)

#nice training loop
def train_with_early_stopping_and_logging(
        model, train_loader, optimizer, scheduler,
        l1_lambda=0.001, l2_lambda=0.001, num_epochs=5, patience=2, clip_value=1.0
):
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    # Lists to store loss values
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = l1_l2_loss(outputs, labels, model, l1_lambda, l2_lambda)
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved the best model')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    return train_losses


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')


# Example usage
#evaluate(model, test_loader)

#Hyperparameter Tuning
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    print(f'\nTesting learning rate: {lr}')
    model = ThreeLayerBiLSTM(vocab_size=len(vocab), embed_size=100, hidden_size=32, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    train(model, train_loader, optimizer, num_epochs=5)
    evaluate(model, test_loader)

# Saving the model
torch.save(model.state_dict(), 'model.pth')
print('Model saved to model.pth')

# Loading the model
loaded_model = ThreeLayerBiLSTM(vocab_size=len(vocab), embed_size=100, hidden_size=128, output_size=2)
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()
print('Model loaded from model.pth')

#---------------------------------------Training process-----------------------------------------------------
# Run the training process with logging
train_losses = train_with_early_stopping_and_logging(model, train_loader, optimizer, scheduler, num_epochs=5)

# Print completion message
print("Training completed.")

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
evaluate(model, test_loader)
