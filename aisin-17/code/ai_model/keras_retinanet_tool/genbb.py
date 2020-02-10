# coding: utf-8

import sys
sys.path.insert(0, '../')
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time


class GenboudingBox:
    """
    This class generate a bound box for image.
    """
    def __init__(self, model_path):
        super(GenboudingBox, self).__init__()
        # adjust this to point to your downloaded/trained model
        # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
        # model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
        # model_path = 'snapshots/resnet50_csv_69.h5'
        # load retinanet model
        self.model = models.load_model(model_path, backbone_name='resnet50')
        # if the model is not converted to an inference model, use the line below
        # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
        self.model = models.convert_model(self.model)
        # load label to names mapping for visualization purposes
        self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'safety cone', 5: 'parking block', 6: 'shopping cart', 7: 'u-shaped barricade', 8: 'short pole', 9: 'traffic light', 10: 'construction barricade', 11: 'traffic sign', 12: 'parking meter', 13: 'tricycle', 14: 'stroller', 15: 'wheelchair', 16: 'other compact ride', 17: 'locked parking barrier', 18: 'unlocked parking barrier'}

    def tagimage(self, image, scale):
        # image = read_image_bgr(image_path)

        # print(image.shape)
        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale
        return boxes, scores, labels
#

# model_path = r'D:\project\retina\models_trained\resnet50_csv_69.h5'
# gen = GenboudingBox(model_path)
# path = r'D:\project\retina\keras_retinanet_tool\examples\FR_20180409113747_0000643.jpeg'
# image = read_image_bgr(path)
# # preprocess image for network
# image = preprocess_image(image)
# image, scale = resize_image(image)
# gen.tagimage(image, scale)
