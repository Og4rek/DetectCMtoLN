import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt


device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 46 * 46, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 46 * 46)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss, train_accuracy = 0, 0
    counter = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        counter += 1
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = loss_fn(output, y)

        train_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_accuracy += (preds == y).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # loss and accuracy for the complete epoch
    epoch_loss = train_loss / counter
    epoch_acc = train_accuracy / len(dataloader.dataset)
    return epoch_acc, epoch_loss


def valid_loop(dataloader, model, loss_fn, epoch, best_valid_acc, folder_path):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, valid_accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            valid_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= num_batches
    valid_accuracy /= size
    print(f"Valid Error: \n Accuracy: {(100*valid_accuracy):>0.1f}%, Avg loss: {valid_loss:>8f} \n")

    if valid_accuracy > best_valid_acc:
        best_valid_acc = valid_accuracy
        print(f"\nBest validation accuracy: {best_valid_acc}")
        print(f"\nSaving best model for epoch: {epoch + 1}\n")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
        }, os.path.join(folder_path, 'best_model.pth'))

    return valid_accuracy, valid_loss, best_valid_acc


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

training_data = datasets.PCAM(root='dataset', split='train', transform=transform)
valid_data = datasets.PCAM(root='dataset', split='val', transform=transform)
test_data = datasets.PCAM(root='dataset', split='test', transform=transform)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = Model()
model.to(device)

print(model)

learning_rate = 1e-3
epochs = 3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_valid_acc = 0

current_datetime = datetime.datetime.now()

year = current_datetime.year
month = current_datetime.month
day = current_datetime.day
hour = current_datetime.hour
minute = current_datetime.minute
second = current_datetime.second

folder_name = f"{year}-{month:02d}-{day:02d}_{hour:02d}-{minute:02d}-{second:02d}"

base_directory = 'outputs'

folder_path = os.path.join(base_directory, folder_name)
os.makedirs(folder_path)

headers = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]

csv_file_path = "history.csv"

print("Training started!")
with open(os.path.join(folder_path, csv_file_path), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        valid_acc, valid_loss, best_valid_acc = valid_loop(valid_dataloader, model, loss_fn, t, best_valid_acc, folder_path)
        writer.writerow([t+1, train_loss, train_acc, valid_loss, valid_acc])
print("Training done!")

print(f"\nSaving last model for epoch: {epochs}\n")
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss_fn,
}, os.path.join(folder_path, 'last_model.pth'))

print("Testing: ")

# load the best model checkpoint
best_model_cp = torch.load(os.path.join(folder_path, 'best_model.pth'))
best_model_epoch = best_model_cp['epoch']
model.load_state_dict(best_model_cp['model_state_dict'])
print(f"Best model was saved at {best_model_epoch} epochs\n")

test_loop(test_dataloader, model, loss_fn)

df = pd.read_csv(os.path.join(folder_path, csv_file_path))

# Create a plot for loss
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['valid_loss'], label='Validation Loss', marker='o')
plt.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(folder_path, "loss_plot.png"))

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['valid_acc'], label='Validation Accuracy', marker='o')
plt.plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(folder_path, "accuracy_plot.png"))

# TODO: 5) Refaktor kodu ladnie na klasy i funkcje podzielic program i tqdm zrobic
# TODO: 6) Stworzyc własny model sieci CNN
# TODO: 7) Stworzyc model sieci GCNN na podstawie CNN
# TODO: 8) Wytrenowac sieć
