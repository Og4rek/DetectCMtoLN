# import os.path
# from PIL import Image
# from src.data.data_loader_h5 import DatasetPCam
#
# dataset_folder = '../../dataset'
# dataset = DatasetPCam(path_to_dataset=dataset_folder)
#
# (trainX, trainY), (testX, testY), (valX, valY) = dataset.get_train(), dataset.get_test(), dataset.get_valid()
#
# for i, (img, label) in enumerate(zip(trainX, trainY)):
#     if not os.path.exists(os.path.join(dataset_folder, 'data_png', 'train')):
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'train'))
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'train', '0'))
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'train', '1'))
#     image_pillow = Image.fromarray(img)
#     image_pillow.save(os.path.join(dataset_folder, 'data_png', 'train', f'{label[0][0][0]}', f'{i:06d}.png'))
#
# for i, (img, label) in enumerate(zip(testX, testY)):
#     if not os.path.exists(os.path.join(dataset_folder, 'data_png', 'test')):
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'test'))
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'test', '0'))
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'test', '1'))
#     image_pillow = Image.fromarray(img)
#     image_pillow.save(os.path.join(dataset_folder, 'data_png', 'test', f'{label[0][0][0]}', f'{i:06d}.png'))
#
# for i, (img, label) in enumerate(zip(valX, valY)):
#     if not os.path.exists(os.path.join(dataset_folder, 'data_png', 'val')):
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'val'))
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'val', '0'))
#         os.makedirs(os.path.join(dataset_folder, 'data_png', 'val', '1'))
#     image_pillow = Image.fromarray(img)
#     image_pillow.save(os.path.join(dataset_folder, 'data_png', 'val', f'{label[0][0][0]}', f'{i:06d}.png'))
