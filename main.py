from src.data.data_loader import DatasetPCam
from src.models.GDenseNet import GDenseNet

dataset_folder = 'dataset'
dataset = DatasetPCam(path_to_dataset=dataset_folder)

model = GDenseNet()

# train model
# test model
