from src.dataset import Dataset
from src.pcammodel import PCAMModel
from att_gconvs.experiments.pcam.models.densenet import *
from att_gconvs.experiments.utils import num_params


def main():
    dataset_folder = 'dataset'
    batch_size = 64

    data_pcam = Dataset(root=dataset_folder, batch_size=batch_size)

    best_models_paths = {
        'DenseNet': '/home/piti/pythonProjects/Magisterka_pytorch/outputs/2024-01-17_11-58-36-model-DenseNet/last_model.pth',
        # 'P4DenseNet': '/home/piti/pythonProjects/Magisterka_pytorch/Wyniki_ostateczne/P4DenseNet/best_model.pth',
        # 'P4MDenseNet': '/home/piti/pythonProjects/Magisterka_pytorch/Wyniki_ostateczne/P4MDenseNet/best_model.pth',
        # 'fA_P4DenseNet': '/home/piti/pythonProjects/Magisterka_pytorch/Wyniki_ostateczne/fA_P4DenseNet/best_model.pth',
        # 'fA_P4MDenseNet': '/home/piti/pythonProjects/Magisterka_pytorch/Wyniki_ostateczne/fA_P4MDenseNet/best_model.pth'
    }

    models = {
        'DenseNet': DenseNet(n_channels=26),
        # 'P4DenseNet': P4DenseNet(n_channels=13),
        # 'P4MDenseNet': P4MDenseNet(n_channels=9),
        # 'fA_P4DenseNet': fA_P4DenseNet(n_channels=13),
        # 'fA_P4MDenseNet': fA_P4MDenseNet(n_channels=9)
    }

    loss_fn = nn.CrossEntropyLoss()

    print("\nTesting: ")
    for model_name, model_path in best_models_paths.items():
        print(model_name)
        load_model = torch.load(model_path)
        models[model_name].load_state_dict(load_model['model_state_dict'])

        pcam_model = PCAMModel(dataset=data_pcam, model=models[model_name], batch_size=batch_size,
                               output_directory=None, lr=None, opt=None, loss=loss_fn, epoch_start=None)

        # print(pcam_model.model)
        print(f"{model_name} parameters: ", end='')
        num_params(models[model_name])

        pcam_model.test()


if __name__ == '__main__':
    main()
