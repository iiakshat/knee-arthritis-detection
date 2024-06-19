from dataclasses import dataclass
from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.utils.helper import read_yaml, create_directories
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import keras

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    def prepare_model(self):
        
        base_model = keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        if self.config.params_include_top:
            model_output = base_model.output
        else:
            flatten_in = keras.layers.Flatten()(base_model.output)
            model_output = flatten_in

        prediction = keras.layers.Dense(
            units=self.config.params_classes,
            activation="softmax"
        )(model_output)

        full_model = keras.models.Model(
            inputs=base_model.input,
            outputs=prediction
        )

        for layer in full_model.layers:
            layer.trainable = False

        optimizer = keras.optimizers.Adam(learning_rate=self.config.params_learning_rate)
        full_model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        self.full_model = full_model
        self.save_model(path=self.config.updated_base_model_path, model=full_model)

    @staticmethod
    def create_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = keras.layers.Flatten()(model.output)
        prediction = keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        full_model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        return full_model

    def update_base_model(self):
        self.full_model = self.create_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path, model):
        model.save(path)