import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import csv
from src.utils import *


class PCAMModel:
    def __init__(self, dataset, model, batch_size, output_directory, lr, opt, loss, epoch_start, k):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dataset = dataset
        self.model = model.to(self.device)
        self.batch_size = batch_size
        if output_directory is not None:
            self.output_path = get_folder_name(output_directory, k)

        # hyperparameters
        self.learning_rate = 1e-2
        self.epochs = 5
        self.epochs_start = epoch_start
        self.loss_fn = loss
        self.optimizer = opt
        if opt is not None:
            self.optimizer_to()
            self.ln_scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-1,
                                                        total_steps=len(dataset.train_dataloader)*self.epochs)
            # self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1),
            #                                                    cooldown=0, patience=10, min_lr=0.5e-6)
        self.early_stopping = {"monitor": 'val_accuracy', "min_delta": 1e-4, "patience": 20}

    def train(self):
        best_valid_acc = 0
        best_valid_acc_es = None
        counter = 0

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
            # print(self.learning_rate)
            valid_acc, valid_loss, best_valid_acc = valid_loop(self.dataset.valid_dataloader,
                                                               self.model,
                                                               self.loss_fn,
                                                               epoch,
                                                               best_valid_acc,
                                                               self.output_path,
                                                               self.device,
                                                               self.optimizer)

            print(f"Train errors: train_loss: {train_loss:>7f}, train_acc: {train_acc:>7f}, "
                  f"valid_loss: {valid_loss:>7f}, valid_acc: {valid_acc:>7f}, lr: {self.optimizer.param_groups[0]['lr']}\n")
            with open(os.path.join(self.output_path, csv_file_path), mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, train_loss, train_acc, valid_loss, valid_acc, self.optimizer.param_groups[0]['lr']])

            # self.lr_scheduler.step(valid_loss)

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

        save_plots(self.output_path, csv_file_path)

    def test(self):
        test_loop(self.dataset.test_dataloader, self.model, self.loss_fn, self.device)

    def optimizer_to(self):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

