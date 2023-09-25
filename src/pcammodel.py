import torch.nn as nn
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
        self.epochs = 5
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        best_valid_acc = 0

        headers = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc"]
        csv_file_path = "history.csv"

        with open(os.path.join(self.output_path, csv_file_path), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")

                train_acc, train_loss = train_loop(self.dataset.train_dataloader,
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

                writer.writerow([epoch + 1, train_loss, train_acc, valid_loss, valid_acc])

        print("Training done!")

        print(f"\nSaving last model for epoch: {self.epochs}\n")
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
