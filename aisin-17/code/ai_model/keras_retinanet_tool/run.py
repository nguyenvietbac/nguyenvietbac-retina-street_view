import os
import numpy as np
from pathlib import Path
import configparser
import tensorflow as tf
from genbb import GenboudingBox
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image, convert_to_big
from keras_retinanet.xmlVOC import make_xml_annotation


def non_maximum_suppression(boxes, scores, labels, num_maxlabels=100):
    out_boxes = []
    out_scores = []
    out_labels = []
    for x in range(num_maxlabels):
        # print(x)
        list_labels = [i for i in range(len(labels)) if labels[i]==x]
        if not list_labels:
            continue
        sco = [scores[i] for i in range(len(scores)) if i in list_labels]
        box = [boxes[i] for i in range(len(boxes)) if i in list_labels]
        select_id = tf.image.non_max_suppression(box, sco, 100, iou_threshold=0.5)
        box = tf.gather(box, select_id)
        sco = tf.gather(sco, select_id)
        out_boxes.extend(box)
        out_scores.extend(sco)
        out_labels.extend([x]*len(sco))
        # print(box)
    out_boxes = np.array(out_boxes)
    out_scores = np.array(out_scores)
    out_labels = np.array(out_labels)
    # print(out_boxes.shape)
    # print(out_scores)
    # print(out_labels)

    return out_boxes, out_scores, out_labels


def tree_output(input_path, output_path, gentool, confident_threshold):
    list_images = os.listdir(input_path)
    # print(list_images)
    for img in list_images:
        # print(img[-4:])
        if os.path.isdir(os.path.join(input_path, img)):
            new_input_path = os.path.join(input_path, img)
            new_output_path = os.path.join(output_path, img)
            if not os.path.isdir(new_output_path):
                os.mkdir(new_output_path)
            tree_output(new_input_path, new_output_path, gentool)
            print(new_input_path, new_output_path)
        elif img[-4:] == 'jpeg' or img[-3:] == 'png':
            img_path = os.path.join(input_path, img)
            image = read_image_bgr(img_path, slide_window=False)
            image1 = read_image_bgr(img_path, slide_window=1)
            image2 = read_image_bgr(img_path, slide_window=2)
            image3 = read_image_bgr(img_path, slide_window=3)

            image = preprocess_image(image)
            image, scale = resize_image(image)
            image1 = preprocess_image(image1)
            image1, scale1 = resize_image(image1)
            image2 = preprocess_image(image2)
            image2, scale2 = resize_image(image2)
            image3 = preprocess_image(image3)
            image3, scale3 = resize_image(image3)

            boxes, scores, labels = gentool.tagimage(image, scale)
            #
            boxes1, scores1, labels1 = gentool.tagimage(image1, scale1)
            boxes1 = convert_to_big(boxes1, type=1)
            boxes2, scores2, labels2 = gentool.tagimage(image2, scale2)
            boxes2 = convert_to_big(boxes2, type=2)
            boxes3, scores3, labels3 = gentool.tagimage(image3, scale3)
            boxes3 = convert_to_big(boxes3, type=3)
            # print(boxes)
            boxes_out = np.concatenate((boxes, boxes1, boxes2, boxes3), axis=1)
            scores_out = np.concatenate((scores, scores1, scores2, scores3), axis=1)
            labels_out = np.concatenate((labels, labels1, labels2, labels3), axis=1)
            print(boxes_out.shape, '\n')
            print(scores_out.shape, '\n')
            print(labels_out.shape, '\n')
            boxes_out, scores_out, labels_out = non_maximum_suppression(boxes_out[0], scores_out[0], labels_out[0])
            # make_xml_annotation(boxes_out[0], scores_out[0], labels_out[0], img, output_path, confident_threshold)
            make_xml_annotation(boxes_out, scores_out, labels_out, img, output_path, confident_threshold)
            # make_xml_annotation(boxes3[0], scores3[0], labels3[0], img, output_path)
            # break


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
    confident_threshold = config['default']['confident_threshold']
    confident_threshold = float(confident_threshold)
    gentool = GenboudingBox(os.path.join(parent_path, 'models_trained', model_name))
    # xml_path = os.path.join(parent_path, output_xml)
    tree_output(input_img, output_xml, gentool, confident_threshold)


if __name__ == '__main__':
    main()
