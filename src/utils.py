import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def train_loop(dataloader, model, loss_fn, optimizer, device, scheduler):
    train_loss, train_accuracy = 0, 0
    counter = 0
    model.train()
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training Progress: "):
        counter += 1
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        train_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    epoch_loss = train_loss / counter
    epoch_acc = train_accuracy / len(dataloader.dataset)

    return epoch_acc, epoch_loss


def valid_loop(dataloader, model, loss_fn, epoch, best_valid_acc, folder_path, device, optimizer, scheduler):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, valid_accuracy = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, total=len(dataloader), desc="Validation Progress: "):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
            valid_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= num_batches
    valid_accuracy /= size

    if valid_accuracy > best_valid_acc:
        best_valid_acc = valid_accuracy
        print(f"Highest validation accuracy find: {best_valid_acc}")
        print(f"Saving best model for epoch: {epoch + 1}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_fn,
        }, os.path.join(folder_path, 'best_model.pth'))

    print(f"Saving last model for epoch: {epoch + 1}")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_fn,
    }, os.path.join(folder_path, 'last_model.pth'))

    return valid_accuracy, valid_loss, best_valid_acc


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    y_true, y_scores = [], []

    with torch.no_grad():
        for X, y in tqdm(dataloader, total=len(dataloader), desc="Testing Progress: "):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_true.extend(y.cpu().numpy())
            y_scores.extend(pred[:, 1].cpu().numpy())  # Assuming binary classification

    test_loss /= num_batches
    correct /= size
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    auc = roc_auc_score(y_true, y_scores)
    print(f"Test done: \n Accuracy: {(100 * correct):>0.2f}%\n Avg loss: {test_loss:>8f}\n AUC: {auc:>0.2f}\n")


def get_folder_name(output_directory, k):
    current_datetime = datetime.datetime.now()

    year = current_datetime.year
    month = current_datetime.month
    day = current_datetime.day
    hour = current_datetime.hour
    minute = current_datetime.minute
    second = current_datetime.second

    folder_name = f"{year}-{month:02d}-{day:02d}_{hour:02d}-{minute:02d}-{second:02d}-model-{k}"

    folder_path = os.path.join(output_directory, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def save_plots(folder_path: str, csv_file_path: str):
    df = pd.read_csv(os.path.join(folder_path, csv_file_path))

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


def save_lr(folder_path: str, csv_file_path: str):
    df = pd.read_csv(os.path.join(folder_path, csv_file_path))

    plt.figure(figsize=(10, 6))
    plt.plot(df['lr'], label='Learning rate', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('Learning rate Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(folder_path, "Learning_rate_plot.png"))
