from src.dataset import Dataset
from src.pcammodel import PCAMModel
from att_gconvs.experiments.pcam.models.densenet import *


def main():
    dataset_folder = 'dataset'
    output_folder = 'outputs'
    batch_size = 64

    data_pcam = Dataset(root=dataset_folder, batch_size=batch_size)

    model = P4DenseNet(n_channels=13)
    model(torch.rand([1, 3, 96, 96]))

    pcam_model = PCAMModel(dataset=data_pcam, model=model, batch_size=batch_size, output_directory=output_folder)

    print(pcam_model.model)
    print("Model parameters: ", end='')
    num_params(model)

    print('\nTraining: ')
    pcam_model.train()

    print("\nTesting: ")
    pcam_model.test()


if __name__ == '__main__':
    main()
