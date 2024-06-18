import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument('--model', type=str, choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'DenseNet', 'P4DenseNet', 'P4MDenseNet', 'fA_P4DenseNet', 'fA_P4MDenseNet'], required=True, help='Model type')
    parser.add_argument('--resnet_freez', action='store_true', help='Freeze ResNet layers during training')
    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--lr', type=str, default=1e-3, help='Learning rate scheduler type')
    parser.add_argument('--lr_pct_start', type=float, default=0.5, help='Starting percentage of learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--epoch', type=int, required=True, help='Number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--continue_train', action='store_true', help='Continue training from the last checkpoint')
    parser.add_argument('--dataset_path', type=str, default="dataset", help='Path to the dataset')
    parser.add_argument('--output_folder', type=str, default="output", help='Folder to save the output')
    parser.add_argument('--log_folder', type=str, default="logs", help='Folder to save the logs')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
