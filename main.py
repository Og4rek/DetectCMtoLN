import argparse
import torch.nn as nn
import torch

from torch.optim import lr_scheduler

from src.pcammodel import PCAMResNetModel
from src.dataset import Dataset


def load_resnet(model_name, freeze_model):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, weights='IMAGENET1K_V1')
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model


def load_densenet():
    return -1


def main(args):
    data_pcam = Dataset(root=args.dataset_path, batch_size=args.batch_size, num_workers=args.num_workers)

    if 'resnet' in args.model:
        model = load_resnet(args.model, args.resnet_freeze)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr / 25)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                            total_steps=len(data_pcam.train_dataloader) * args.epoch)

        epoch_start = 0
        if len(args.model_path) > 0:
            load_model = torch.load(args.model_path)
            model.load_state_dict(load_model['model_state_dict'])
            optimizer.load_state_dict(load_model['optimizer_state_dict'])
            scheduler.load_state_dict(load_model['scheduler_state_dict'])
            epoch_start = load_model['epoch']

        loss_fn = nn.CrossEntropyLoss()

        pcam_model = PCAMResNetModel(dataset=data_pcam, model=model, batch_size=args.batch_size,
                                     output_directory=args.output_folder, max_lr=args.max_lr, epochs=args.epoch,
                                     opt=optimizer, loss=loss_fn, epoch_start=epoch_start,
                                     scheduler=scheduler, k=f'{args.model}_{args.resnet_freeze}')
    else:
        pcam_model = load_densenet()

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

    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'DenseNet', 'P4DenseNet','P4MDenseNet','fA_P4DenseNet', 'fA_P4MDenseNet'], required=True, help='Model type')
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
    parser.add_argument('--test_model', type=str, daction='store_true', help='Only test given model with out training')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
