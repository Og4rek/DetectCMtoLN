import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import csv
from src.utils import *


class PCAMModel:
    def __init__(self, dataset, model, batch_size, output_directory):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = dataset
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.output_path = get_folder_name(output_directory)

        # hyperparameters
        self.learning_rate = 1e-3
        self.epochs = 2
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1), cooldown=0,
                                                           patience=10, min_lr=0.5e-6)
        self.early_stopping = {"monitor": 'val_accuracy', "min_delta": 1e-4, "patience": 20}

    def train(self):
        best_valid_acc = 0
        best_valid_acc_es = None
        counter = 0

        headers = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
        csv_file_path = "history.csv"

        with open(os.path.join(self.output_path, csv_file_path), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")

                train_acc, train_loss = train_loop(self.dataset.test_dataloader,
                                                   self.model,
                                                   self.loss_fn,
                                                   self.optimizer,
                                                   self.device)

                valid_acc, valid_loss, best_valid_acc = valid_loop(self.dataset.valid_dataloader,
                                                                   self.model,
                                                                   self.loss_fn,
                                                                   epoch,
                                                                   best_valid_acc,
                                                                   self.output_path,
                                                                   self.device,
                                                                   self.optimizer)

                print(f"Train errors: train_loss: {train_loss:>7f}, train_acc: {train_acc:>7f}, "
                      f"valid_loss: {valid_loss:>7f}, valid_acc: {valid_loss:>7f}\n")
                writer.writerow([epoch + 1, train_loss, train_acc, valid_loss, valid_acc])

                self.lr_scheduler.step(valid_loss)

                if best_valid_acc_es is None or valid_acc > best_valid_acc_es + self.early_stopping["min_delta"]:
                    best_valid_acc_es = valid_acc
                    counter = 0
                else:
                    counter += 1

                if counter >= self.early_stopping["patience"]:
                    print("Early stopping due to lack of improvement in validation accuracy")
                    break

        print("Training done!")

        print(f"Saving last model for epoch: {self.epochs}")
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn,
        }, os.path.join(self.output_path, 'last_model.pth'))

    def test(self):
        best_model = torch.load(os.path.join(self.output_path, 'best_model.pth'))
        self.model.load_state_dict(best_model['model_state_dict'])
        test_loop(self.dataset.test_dataloader, self.model, self.loss_fn, self.device)
