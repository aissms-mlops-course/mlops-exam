import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from glob import glob
import os
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf 


def train_model(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['image_size']
        train_set = config['model']['train_path']
        test_set = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertical_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch_size = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']
        model_path = config['model']['sav_dir']

        print(type(batch_size))

        resnet = VGG16(weights='imagenet', include_top=False, input_shape = img_size + [3])
        for p in resnet.layers:
            p.trainable = False
        op = Flatten()(resnet.output)
        prediction = Dense(num_cls, activation='softmax')(op)
        mod = Model(inputs = resnet.input, outputs = prediction)
        print(mod.summary())

        mod.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        train_gen = ImageDataGenerator(rescale=rescale, 
                                       shear_range=shear_range, 
                                       zoom_range=zoom_range, 
                                       horizontal_flip=horizontal_flip, 
                                       vertical_flip=vertical_flip, 
                                       rotation_range=90)
        test_gen = ImageDataGenerator(rescale=rescale)

        train_set =  train_gen.flow_from_directory(train_set, 
                                                   target_size=img_size, 
                                                   batch_size=batch_size, 
                                                   class_mode=class_mode)
        test_set = test_gen.flow_from_directory(test_set, 
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode=class_mode)

        history = mod.fit(train_set,
                          epochs = epochs,
                          validation_data = test_set,
                          steps_per_epoch=len(train_set),
                          validation_steps = len(test_set)) 

        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.plot(history.history['accuracy'], label='training accuracy')
        plt.plot(history.history['val_accuracy'], label = 'validation accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy') 
        plt.legend()
        plt.savefig('reports/model_train.png')
        plt.show()

        mod.save(model_path)
        print("Model saved at {}".format(model_path))

    else:
        print("Model is not trainable") 

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    a = train_model(config_file=parsed_args.config)
