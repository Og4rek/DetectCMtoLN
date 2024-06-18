import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import csv
from src.utils import *


class PCAMModel:
    def __init__(self, dataset, model, batch_size, output_directory, epochs, opt, loss, epoch_start, scheduler,
                 checkpoint_folder):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = dataset
        self.model = model.to(self.device)
        self.batch_size = batch_size
        if output_directory is not None:
            self.output_path = get_folder_name(output_directory, checkpoint_folder)

        # hyperparameters
        self.epochs = epochs
        self.epochs_start = epoch_start
        self.loss_fn = loss
        self.optimizer = opt
        self.ln_scheduler = scheduler

    def train(self):
        best_valid_acc = 0

        headers = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "lr"]
        csv_file_path = "history.csv"

        with open(os.path.join(self.output_path, csv_file_path), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        for epoch in range(self.epochs_start, self.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")

            train_acc, train_loss = train_loop(self.dataset.train_dataloader,
                                               self.model,
                                               self.loss_fn,
                                               self.optimizer,
                                               self.device,
                                               self.ln_scheduler)

            valid_acc, valid_loss, best_valid_acc = valid_loop(self.dataset.valid_dataloader,
                                                               self.model,
                                                               self.loss_fn,
                                                               epoch,
                                                               best_valid_acc,
                                                               self.output_path,
                                                               self.device,
                                                               self.optimizer,
                                                               self.ln_scheduler)

            if isinstance(self.ln_scheduler, ReduceLROnPlateau):
                self.ln_scheduler.step(valid_loss)

            print(f"Train errors: train_loss: {train_loss:>7f}, train_acc: {train_acc:>7f}, "
                  f"valid_loss: {valid_loss:>7f}, valid_acc: {valid_acc:>7f}, lr: {self.optimizer.param_groups[0]['lr']}\n")
            with open(os.path.join(self.output_path, csv_file_path), mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, train_loss, train_acc, valid_loss, valid_acc, self.optimizer.param_groups[0]['lr']])

        print("Training done!")

        save_plots(self.output_path, csv_file_path)

    def test(self):
        test_loop(self.dataset.test_dataloader, self.model, self.loss_fn, self.device)
