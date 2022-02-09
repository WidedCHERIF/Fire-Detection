#Importing libraries
import numpy as np
from numpy import loadtxt
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, cohen_kappa_score,classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers, applications
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Flatten,GlobalMaxPooling2D
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import metrics


#Specifying Important parameters
bacth_size = 16
epochs = 50
warmup_epocks = 2
learning_rate = 0.00001
warmup_learning_rate = 0.00008
height = 128
width = 128
colors = 3
n_classes = 2
es_patience = 18
rlrop_patience = 3
decay_drop = 0.5
based_model_last_block_layer_number = 0

#Import data
train_dir = 'fire_dataset/train/'
validation_dir = 'fire_dataset/test/'


#Data generation
train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.5,
      brightness_range=[0.7,1.3],
      horizontal_flip=True,
      fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        batch_size= bacth_size,
        shuffle = True,
        class_mode= 'categorical')

val_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size = bacth_size,
        shuffle=True,
        class_mode= 'categorical')

#Creating the Model
def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.ResNet50(weights='imagenet',  
                                        include_top=False,
                                        input_tensor=input_tensor)
    print(base_model.summary())
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(1024, activation='relu', name='output')(x) 
    output = Dropout(0.5)(output)

    model_prim = Model(input_tensor, output)
    final_output = Dense(n_out, activation='softmax', name='final_output')(model_prim.output)
    model = Model(input_tensor, final_output)

    return model

model = create_model(input_shape=(height, width, colors), n_out=n_classes)

#indicates which layers will be trained
for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

#Optimiser and Loss function
metric_list=['accuracy']
optimizer = optimizers.Adam(lr=warmup_learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

#reduce the learning rate progressively when needed
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=rlrop_patience, factor=decay_drop, min_lr=1e-6, verbose=1)


#Model Evaluation
metric_list=['accuracy']
optimizer = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)
print(model.summary())


#generating the number of iteration in every epoch
step_train = train_generator.n//train_generator.batch_size
step_validation = val_generator.n//val_generator.batch_size
print(train_generator.n)

#saving the model
checkpointer = ModelCheckpoint(filepath='model.h5',monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

callback_list = [rlrop,checkpointer]

history_warmup = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_train,
                              validation_data=val_generator,
                              validation_steps=step_validation,
                              callbacks=[checkpointer],
                              epochs=1,
                              verbose=1).history



history = model.fit_generator(generator=train_generator,
                             steps_per_epoch=step_train,
                             validation_data=val_generator,
                              validation_steps=step_validation,
                              epochs=epochs,
                              callbacks=callback_list,
                              verbose=1).history



model.save('modell.h5')