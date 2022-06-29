from preprocessing import Preprocessing
import model_cd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os

class Train:
    def __init__(self, data_root=''):
        self.pp = Preprocessing(data_root='datasets')
        self.model_cls = model_cd.ModelCatDog()

    def training(self):
        train_ds, val_ds = self.pp.get_train_val_ds()

        model = self.model_cls.get_model()
        model_checkpoint = ModelCheckpoint(
            filepath='weights/best_inception_v3_cat_dog_1000.h5')
        tensor_board = TensorBoard(log_dir='logs')
        model.fit_generator(train_ds, validation_data=val_ds, epochs=1000,
                            callbacks=[model_checkpoint, tensor_board])


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # if you want to run on GPU then comment this line

    train_cls = Train()
    train_cls.training()
