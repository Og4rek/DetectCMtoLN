import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

from src.models.GDenseNet import GDenseNet
from keras import backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

dataset_folder = 'dataset/data_png'
# dataset = DatasetPCam(path_to_dataset=dataset_folder)

batch_size = 16
nb_classes = 2
epochs = 1

img_rows, img_cols = 96, 96
img_channels = 3
image_size = (96, 96)

# Tworzenie generatorów danych
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    f'{dataset_folder}/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)
train_size = train_generator.n

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow_from_directory(
    f'{dataset_folder}/val',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)
val_size = val_generator.n

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    f'{dataset_folder}/test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)
test_size = test_generator.n


def custom_generator(generator):
    for batch_x, batch_y in generator:
        one_hot_labels = tf.one_hot(batch_y, depth=nb_classes)
        yield batch_x, one_hot_labels


train_generator = custom_generator(train_generator)
val_generator = custom_generator(val_generator)
test_generator = custom_generator(test_generator)

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

batch_images = next(train_generator)
one_image = batch_images[0][0]

# Test equivariance by comparing outputs for rotated versions of same datapoint:
res = model.predict(np.stack([one_image, np.rot90(one_image)]))
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

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_size // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    verbose=1
)

scores = model.evaluate_generator(
    generator=test_generator,
    batch_size=batch_size,
)

print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])

# TODO: odpalic na gpu
