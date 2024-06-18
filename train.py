from src.pcammodel import PCAMResNetModel
from src.dataset import Dataset
from torch.optim import lr_scheduler
import torch.nn as nn
import torch

dataset_folder = 'dataset'
output_folder = 'outputs'

# hyperparameters
batch_size = 64
max_learning_rate = 1e-3
learning_rate = max_learning_rate / 25
loss_fn = nn.CrossEntropyLoss()
epochs = 10

data_pcam = Dataset(root=dataset_folder, batch_size=batch_size)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights='IMAGENET1K_V2')

# for param in model.parameters():
#     param.requires_grad = False

num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.to('cuda')

epoch_start = 0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=len(data_pcam.train_dataloader)*epochs)

model_continue = '/home/piti/python_projects/magisterka/DetectCMtoLN/outputs/2024-06-04_16-16-17-model-resnet101_unfreeze/last_model.pth'
if len(model_continue) > 0:
    load_model = torch.load(model_continue)
    model.load_state_dict(load_model['model_state_dict'])
    optimizer.load_state_dict(load_model['optimizer_state_dict'])
    scheduler.load_state_dict(load_model['scheduler_state_dict'])
    epoch_start = load_model['epoch']

pcam_model = PCAMResNetModel(dataset=data_pcam, model=model, batch_size=batch_size, output_directory=output_folder,
                             max_lr=max_learning_rate, epochs=epochs, opt=optimizer, loss=loss_fn,
                             epoch_start=epoch_start, scheduler=scheduler, k='resnet101_unfreeze')

# print(pcam_model.model)

# Calculate trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", trainable_params)

print('\nTraining: ')
pcam_model.train()

print("\nTesting: ")
pcam_model.test()


# resnet: unfreez, freez: 18, 34, 50, 101, 152
        # 'DenseNet': DenseNet(n_channels=26),
        # 'P4DenseNet': P4DenseNet(n_channels=13),
        # 'P4MDenseNet': P4MDenseNet(n_channels=9),
        # 'fA_P4DenseNet': fA_P4DenseNet(n_channels=13),
        # 'fA_P4MDenseNet': fA_P4MDenseNet(n_channels=9)