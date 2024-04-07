from src.dataset import Dataset
from src.pcammodel import PCAMModel
from att_gconvs.experiments.pcam.models.densenet import *
from att_gconvs.experiments.utils import num_params


def main():
    dataset_folder = 'dataset'
    output_folder = 'outputs'
    batch_size = 64

    data_pcam = Dataset(root=dataset_folder, batch_size=batch_size)

    print()

    models = {
        'DenseNet': DenseNet(n_channels=26),
        'P4DenseNet': P4DenseNet(n_channels=13),
        'P4MDenseNet': P4MDenseNet(n_channels=9),
        'fA_P4DenseNet': fA_P4DenseNet(n_channels=13),
        'fA_P4MDenseNet': fA_P4MDenseNet(n_channels=9)
    }

    # for k, v in models.items():
    learning_rate = 1e-3
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False)
    # num_classes = 2
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = fA_P4MDenseNet(n_channels=9)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_start = 0
    # epoch_start = 0
    # model_continue = ''
    model_continue = '/kaggle/input/pcamfap4m/pytorch/v1/1/last_model.pth'
    if len(model_continue) > 0:
        load_model = torch.load(model_continue)
        model.load_state_dict(load_model['model_state_dict'])
        optimizer.load_state_dict(load_model['optimizer_state_dict'])
        epoch_start = load_model['epoch']
    #
    # model(torch.rand([1, 3, 96, 96]))
    #
    pcam_model = PCAMModel(dataset=data_pcam, model=model, batch_size=batch_size, output_directory=output_folder,
                           lr=learning_rate, opt=optimizer, loss=loss_fn, epoch_start=epoch_start, k='fA_P4MDenseNet_aug')
    #

    print(pcam_model.model)
    print("Model parameters: ", end='')
    num_params(model)

    print(pcam_model.device)

    print('\nTraining: ')
    pcam_model.train()

    print("\nTesting: ")
    pcam_model.test()

    # model_continue = ''


if __name__ == '__main__':
    main()
