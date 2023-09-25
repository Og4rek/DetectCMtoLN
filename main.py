from src.dataset import Dataset
from src.model import Model
from src.pcammodel import PCAMModel


def main():
    dataset_folder = 'dataset'
    output_folder = 'outputs'
    batch_size = 64

    data_pcam = Dataset(root=dataset_folder, batch_size=batch_size)
    model = Model()
    pcam_model = PCAMModel(dataset=data_pcam, model=model, batch_size=batch_size, output_directory=output_folder)

    print(pcam_model.model)

    print('\nTraining: ')
    pcam_model.train()

    print("\nTesting: ")
    pcam_model.test()


if __name__ == '__main__':
    main()
