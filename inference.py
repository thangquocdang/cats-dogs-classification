
import cv2 as cv
import model_cd
import numpy as np
import os

class InferenceCatDog:
    def __init__(self):
        weight_path = r"cat_dog_cls\weights\best_inception_v3_cat_dog_1000.h5"
        self.model_input_size = (160,160)
        model_cls = model_cd.ModelCatDog()
        self.model = model_cls.get_model()
        self.model.load_weights(weight_path)

    def predict(self, image):
        image = cv.resize(image, self.model_input_size)
        image = image / 255.0
        imgs = np.array([image])
        y_pred = self.model.predict(imgs)
        return y_pred  


    def predict_image_path(self, image_path):
        img = cv.imread(image_path)
        self.predict(image= img)
    

if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # if you want to run on GPU then comment this line
    pred = InferenceCatDog()

    out = pred.predict_image_path(image_path=r"cat_dog_cls\datasets\validation\dogs\dog.2004.jpg")
    print(out)

    if out[0][0] > 0.5:
        print('Dog')
    else:
        print('Cat')