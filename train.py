import os
import matplotlib.pyplot as plt
import numpy as np
import random 
import cv2
from data_utils import splite_train_valid, DataGenerator
from model import resnet_18
import tensorflow as tf
import shutil
import tensorflow_addons as tfa

for f in os.listdir('dataset/backup/'):
    shutil.copy('dataset/backup/'+f, 'dataset/digital_3d_processed/')
shutil.rmtree('Train')
shutil.rmtree('Valid')

dataset_dir = 'dataset/digital_3d_processed/'
train_list,valid_list = splite_train_valid(dataset_dir, 0.7)

train_datagen = DataGenerator(train_list)
valid_datagen = DataGenerator(valid_list,mode='testing')

num_classes = 10
input_shape = (256, 256, 1)


def create_encoder():
    resnet = resnet_18(input_shape=input_shape)
    inputs = tf.keras.Input(shape=input_shape)
    outputs = resnet(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10-encoder")
    return model

input_shape = (256, 256, 1)
learning_rate = 0.001
batch_size = 64
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    features = encoder(inputs)
    #features = tf.keras.layers.Dropout(dropout_rate)(features)
    #features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    #features = tf.keras.layers.Dropout(dropout_rate)(features)
    #outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)

    model = tf.keras.Model(inputs=inputs, outputs=features, name="cifar10-classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = resnet_18(input_shape=input_shape,learning_rate=learning_rate,dropout_rate=dropout_rate,num_class=num_classes,num_hidden=hidden_units)



my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00001)
]



history = model.fit_generator(train_datagen,epochs=num_epochs,steps_per_epoch=100,validation_data=valid_datagen,validation_steps=200,verbose=1,callbacks=my_callbacks)

