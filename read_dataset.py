import h5py
import numpy as np
import matplotlib.pyplot as plt


def read_train_dataset():
    # Open the HDF5 file
    file_path_x = "dataset/camelyonpatch_level_2_split_train_x.h5"
    file_path_y = "dataset/camelyonpatch_level_2_split_train_y.h5"
    h5_file_x = h5py.File(file_path_x, "r")
    h5_file_y = h5py.File(file_path_y, "r")

    dataset_x = h5_file_x['x']
    dataset_y = h5_file_y['y']

    # Read the data into a NumPy array
    x_train = dataset_x[:]
    y_train = dataset_y[:]

    # Close the HDF5 file
    h5_file_x.close()
    h5_file_y.close()

    return x_train, y_train


x_train, y_train = read_train_dataset()
print(f"Shape of x_train and y_train:  {x_train.shape}, {y_train.shape}")

num_samples = x_train.shape[0]
indices = np.random.choice(num_samples, size=16, replace=False)

plt.figure(figsize=(10, 10))

cancer_labels = ['No', 'Yes']

for i, index in enumerate(indices):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[index])
    plt.axis('off')
    plt.title(f"Is cancer: {cancer_labels[y_train[i, 0, 0, 0]]}")

plt.show()
