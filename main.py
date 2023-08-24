from src.data.data_loader import DatasetPCam
from src.models.GDenseNet import GDenseNet
from keras import backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils import np_utils

dataset_folder = 'dataset'
dataset = DatasetPCam(path_to_dataset=dataset_folder)

batch_size = 1
nb_classes = 2
epochs = 1

img_rows, img_cols = 96, 96
img_channels = 3

# Parameters for the DenseNet model builder
img_dim = (img_channels, img_rows, img_cols) if backend.image_data_format() == 'channels_first' else (
    img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
nb_filter = 16
dropout_rate = 0.0  # 0.0 for data augmentation
conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
use_gcnn = True

# Create the model (without loading weights)
model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                  nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
                  use_gcnn=use_gcnn, conv_group=conv_group, classes=nb_classes)
print('Model created')

model.summary()

optimizer = Adam(learning_rate=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print('Finished compiling')

(trainX, trainY), (testX, testY), (valX, valY) = dataset.get_train(), dataset.get_test(), dataset.get_valid()
X_train = trainX[:100].astype('float16')
X_test = testX[:100].astype('float16')
X_val = valX[:100].astype('float16')

X_train /= 255.
X_test /= 255.
X_val /= 255.

Y_train = np_utils.to_categorical(trainY, nb_classes)[:100, 0, 0]
Y_test = np_utils.to_categorical(testY, nb_classes)[:100, 0, 0]
Y_val = np_utils.to_categorical(testY, nb_classes)[:100, 0, 0]

# Test equivariance by comparing outputs for rotated versions of same datapoint:
res = model.predict(np.stack([trainX[23], np.rot90(trainX[23])]))
is_equivariant = np.allclose(res[0], res[1])
print('Equivariance check:', is_equivariant)
assert is_equivariant

weights_file = 'DenseNet.h5'

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')

callbacks = [lr_reducer, early_stopper, model_checkpoint]

model.fit(X_train, Y_train,
          batch_size=batch_size,
          steps_per_epoch=len(X_train) // batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(X_val, Y_val),
          verbose=1)

scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])
