import h5py
import os
import glob


class DatasetPCam:
    def __init__(self, path_to_dataset="dataset"):
        self.path_to_dataset = path_to_dataset

        self._train_x = None
        self._train_y = None
        self._test_x = None
        self._test_y = None
        self._valid_x = None
        self._valid_y = None

        self.read_dataset()

    def read_dataset(self):
        paths_to_files = self.get_filenames_from_folder()
        set_data = {"test": {"x": self.set_test_x, "y": self.set_test_y},
                    "train": {"x": self.set_train_x, "y": self.set_train_y},
                    "valid": {"x": self.set_valid_x, "y": self.set_valid_y},
                    }
        for file_path in paths_to_files:
            filename = os.path.basename(file_path)
            split_type, data_type = filename.split('_')[4], filename.split('.')[0][-1]  # images: x or labels: y
            data = self.get_h5datafile_as_numpy_matrix(file_path, data_type)
            set_data[split_type][data_type](data)

    def get_filenames_from_folder(self):
        return glob.glob(os.path.join(self.path_to_dataset, "*.h5"))

    def get_h5datafile_as_numpy_matrix(self, file_path, data_type):
        file = self.get_h5datafile(file_path)
        data = file[data_type]
        data_arr = data[:]
        file.close()
        return data_arr

    @staticmethod
    def get_h5datafile(path_to_file):
        return h5py.File(path_to_file, "r")

    def get_train_x(self):
        return self._train_x

    def set_train_x(self, data):
        self._train_x = data

    def get_train_y(self):
        return self._train_y

    def set_train_y(self, data):
        self._train_y = data

    def get_train(self):
        return self.get_train_x(), self.get_train_y()

    def get_test_x(self):
        return self._train_x

    def set_test_x(self, data):
        self._test_x = data

    def get_test_y(self):
        return self._train_y

    def set_test_y(self, data):
        self._test_y = data

    def get_test(self):
        return self.get_test_x(), self.get_test_y()

    def get_valid_x(self):
        return self._train_x

    def set_valid_x(self, data):
        self._valid_x = data

    def get_valid_y(self):
        return self._train_y

    def set_valid_y(self, data):
        self._valid_y = data

    def get_valid(self):
        return self.get_valid_x(), self.get_valid_y()


