
import tensorflow as tf


class Preprocessing:
    def __init__(self, data_root = ''):
        self.train_path = f"{data_root}/train"
        self.val_path = f"{data_root}/validation"
        self.image_size_input_model = (160, 160)
        self.class_mode = 'binary'
        self.batch_size = 8

    def get_train_val_ds(self):
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255.,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)
        val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255.)

        train_ds = train_gen.flow_from_directory(
            directory=self.train_path,
            target_size=self.image_size_input_model,
            class_mode=self.class_mode,
            batch_size=self.batch_size)
        val_ds = val_gen.flow_from_directory(
            directory=self.val_path,
            target_size=self.image_size_input_model,
            class_mode=self.class_mode,
            batch_size=self.batch_size)

        return train_ds, val_ds


if __name__ == "__main__":
    pp = Preprocessing()
    train_ds, val_ds = pp.get_train_ds()
