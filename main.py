import argparse
import torch.nn as nn
import torch

from torch.optim import lr_scheduler

from src.pcam_model import PCAMModel
from src.dataset import Dataset

import os
import sys

try:
    cwd = os.getcwd()
    sys.path.append(os.getcwd()+'/att_gconvs')
    from experiments.pcam.models.densenet import DenseNet, P4DenseNet, P4MDenseNet, fA_P4DenseNet, fA_P4MDenseNet
except ModuleNotFoundError:
    print(f'Module att_gconvs not found!')
    sys.exit(1)


def load_checkpoint(model_path, model, optimizer, scheduler):
    load_model = torch.load(model_path)
    model.load_state_dict(load_model['model_state_dict'])
    optimizer.load_state_dict(load_model['optimizer_state_dict'])
    scheduler.load_state_dict(load_model['scheduler_state_dict'])
    epoch_start = load_model['epoch']

    return model, optimizer, scheduler, epoch_start


def load_pcam_resnet(args, model, dataset):
    model_freeze = 'freeze' if args.resnet_freeze else 'unfreeze'
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr / 25)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                        total_steps=len(dataset.train_dataloader) * args.epoch)

    epoch_start = 0
    if len(args.model_path) > 0:
        model, optimizer, scheduler, epoch_start = load_checkpoint(args.model_path, model, optimizer, scheduler)

    pcam_model = PCAMModel(dataset=dataset, model=model, batch_size=args.batch_size,
                           output_directory=args.output_folder, epochs=args.epoch, opt=optimizer, loss=loss_fn,
                           epoch_start=epoch_start, scheduler=scheduler,
                           checkpoint_folder=f'{args.model}_{model_freeze}')

    return pcam_model


def load_resnet(args, dataset):
    model = torch.hub.load('pytorch/vision:v0.10.0', args.model, weights='IMAGENET1K_V1')
    if args.resnet_freeze:
        for param in model.parameters():
            param.requires_grad = False

    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return load_pcam_resnet(args, model, dataset)


def load_pcam_densenet(args, model, dataset):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    epoch_start = 0
    model_continue = ''
    if len(model_continue) > 0:
        load_model = torch.load(model_continue)
        model.load_state_dict(load_model['model_state_dict'])
        optimizer.load_state_dict(load_model['optimizer_state_dict'])
        epoch_start = load_model['epoch']

    pcam_model = PCAMModel(dataset=dataset, model=model, batch_size=args.batch_size,
                           output_directory=args.output_folder, epochs=args.epoch, opt=optimizer, loss=loss_fn,
                           epoch_start=epoch_start, scheduler=scheduler, checkpoint_folder=args.model)

    return pcam_model


def load_densenet(args, dataset):
    match args.model:
        case 'DenseNet':
            model = DenseNet(n_channels=26)
        case 'P4DenseNet':
            model = P4DenseNet(n_channels=13)
        case 'P4MDenseNet':
            model = P4MDenseNet(n_channels=9)
        case 'fA_P4DenseNet':
            model = fA_P4DenseNet(n_channels=13)
        case 'fA_P4DenseNet':
            model = fA_P4MDenseNet(n_channels=9)
        case _:
            raise Exception('Invalid model name')

    return load_pcam_densenet(args, model, dataset)


def main(args):
    pcam_dataset = Dataset(root=args.dataset_path, batch_size=args.batch_size, num_workers=args.num_workers,
                           augumentation=args.aug)

    if 'resnet' in args.model:
        pcam_model = load_resnet(args, pcam_dataset)
    else:
        pcam_model = load_densenet(args, pcam_dataset)

    print(pcam_model.model)

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in pcam_model.model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)

    if args.test_model:
        print("\nTesting: ")
        pcam_model.test()
    else:
        print('\nTraining: ')
        pcam_model.train()

        print("\nTesting: ")
        pcam_model.test()


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'DenseNet', 'P4DenseNet',
                                 'P4MDenseNet', 'fA_P4DenseNet', 'fA_P4MDenseNet'], required=True, help='Model type')
    parser.add_argument('--resnet_freeze', action='store_true', help='Freeze ResNet layers during training')
    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--lr', type=str, default=1e-3, help='Learning rate scheduler type')
    parser.add_argument('--lr_pct_start', type=float, default=0.5, help='Starting percentage of learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--epoch', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--model_path', default='', help='Continue training from the given checkpoint or test model')
    parser.add_argument('--dataset_path', type=str, default="dataset", help='Path to the dataset')
    parser.add_argument('--output_folder', type=str, default="output", help='Folder to save the output')
    parser.add_argument('--log_folder', type=str, default="logs", help='Folder to save the logs')
    parser.add_argument('--test_model', action='store_true', help='Only test given model without training')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
