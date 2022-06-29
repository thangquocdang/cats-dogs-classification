from pickletools import optimize
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model

class ModelCatDog:
    def __init__(self):
        self.mode_input_size = (160, 160, 3)

    def get_model(self):
        model_inception_v3 = InceptionV3(
            include_top=False, weights='imagenet', input_shape=self.mode_input_size)
        for layer in model_inception_v3.layers:
            layer.trainable = False

        inceptionV3_nt_out_layer = model_inception_v3.get_layer('mixed10')
        inceptionV3_nt_out = inceptionV3_nt_out_layer.output

        x = Flatten()(inceptionV3_nt_out)
        x = Dense(1024)(x)
        x = relu(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(512)(x)
        x = relu(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(256)(x)
        x = relu(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(128)(x)
        x = relu(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        x = Dense(1, activation='sigmoid')(x)
        
        model = Model(model_inception_v3.input, x)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=['accuracy'])

        return model


if __name__ == "__main__":
    model_cd_cls = ModelCatDog()
    model = model_cd_cls.get_model()
    model.summary()