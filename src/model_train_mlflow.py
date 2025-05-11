import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf 
import mlflow
from urllib.parse import urlparse
import mlflow.keras

def train_model_mlflow(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    
    if train:
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
        model_path = config['model_mlflow_config']['model_mlfow']

        print(f"Batch size: {type(batch_size)}")

        # Use VGG16 as a base model with pre-trained weights (without the top layer)
        resnet = VGG16(weights='imagenet', include_top=False, input_shape=img_size + [3])
        for p in resnet.layers:
            p.trainable = False

        # Add custom layers on top of the VGG16 base model
        op = Flatten()(resnet.output)
        prediction = Dense(num_cls, activation='softmax')(op)
        mod = Model(inputs=resnet.input, outputs=prediction)
        print(mod.summary())

        mod.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Data Augmentation for training and testing
        train_gen = ImageDataGenerator(rescale=rescale,
                                       shear_range=shear_range,
                                       zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip,
                                       vertical_flip=vertical_flip,
                                       rotation_range=90)
        test_gen = ImageDataGenerator(rescale=rescale)

        train_set = train_gen.flow_from_directory(train_set,
                                                  target_size=img_size,
                                                  batch_size=batch_size,
                                                  class_mode=class_mode)
        test_set = test_gen.flow_from_directory(test_set,
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode=class_mode)

        # MLflow Setup
        mlflow_config = config['mlflow_config']
        remote_server_uri = mlflow_config['remote_server_uri']
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(mlflow_config['experiment_name'])

        with mlflow.start_run():
            # Training the model
            history = mod.fit(train_set,
                              epochs=epochs,
                              validation_data=test_set,
                              steps_per_epoch=len(train_set),
                              validation_steps=len(test_set))

            # Log metrics to MLflow
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]
            accuracy = history.history['accuracy'][-1]

            # Logging parameters and metrics
            mlflow.log_param('epochs', epochs)
            mlflow.log_param('loss', loss)
            mlflow.log_param('val_loss', val_loss)
            mlflow.log_param('val_accuracy', val_acc)
            mlflow.log_param('metrics', accuracy)

            # Log the model to MLflow
            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(mod, "model", registered_model_name=mlflow_config['registered_model_name'])
            else:
                mlflow.keras.log_model(mod, "model")

            # Save model locally
            mod.save(model_path)
            print(f"Model saved at {model_path}")

            # Save training plot
            plt.plot(history.history['loss'], label='training loss')
            plt.plot(history.history['val_loss'], label='validation loss')
            plt.plot(history.history['accuracy'], label='training accuracy')
            plt.plot(history.history['val_accuracy'], label='validation accuracy')

            plt.xlabel('Epochs')
            plt.ylabel('Loss/Accuracy')
            plt.legend()
            plt.savefig('reports/model_train.png')
            plt.show()

    else:
        print("Model is not trainable")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_model_mlflow(config_file=parsed_args.config)
