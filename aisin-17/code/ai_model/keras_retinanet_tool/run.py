import sys
import os
# sys.path.append(os.getcwd() + '/..')
from pathlib import Path
import cv2
import configparser
import matplotlib.pyplot as plt
from genbb import GenboudingBox
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.xmlVOC import make_xml_annotation

def main():
    # Read constant
    config = configparser.ConfigParser()
    cwd = os.path.dirname(os.path.realpath(__file__))
    path = Path(cwd)
    parent_path = path.parent
    config.read(os.path.join(parent_path, 'constant.ini'))
    input_img = config['default']['data_path']
    output_xml = config['default']['result_path']
    model_name = config['default']['model_file']
    gentool = GenboudingBox(os.path.join(parent_path, 'models_trained', model_name))
    list_images = os.listdir(input_img)
    xml_path = os.path.join(parent_path, output_xml)
    # print(list_images)
    for img in list_images:
        if img[-4:] != 'jpeg' and img[-3:] != 'png':
            continue
        img_path = os.path.join(input_img, img)
        # print(img_path)
        image = read_image_bgr(img_path, slide_window=False)
        # image1 = read_image_bgr(img_path, slide_window=1)
        # image2 = read_image_bgr(img_path, slide_window=2)
        # image3 = read_image_bgr(img_path, slide_window=3)

        image = preprocess_image(image)
        image, scale = resize_image(image)

        boxes, scores, labels = gentool.tagimage(image, scale)
        # boxes1, scores1, labels1 = gentool.tagimage(image1)
        # boxes2, scores2, labels2 = gentool.tagimage(image2)
        # boxes3, scores3, labels3 = gentool.tagimage(image3)
        # print(boxes)
        # print(scores)
        # print(labels)
        make_xml_annotation(boxes[0], scores[0], labels[0], img, xml_path)
        # break


if __name__ == '__main__':
    main()
