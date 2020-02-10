import cv2
import os
import xml.etree.ElementTree as ET
import pandas as pd

##############################
# img = cv2.imread("examples/FR_20190215112728_0000196.jpeg")
# img = cv2.imread("examples/FR_20190214140417_0000082.jpeg")


def parsexml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_name = []
    filename = root.find('filename').text
    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        count = 0
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
        key = boxes.find("name").text
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        # print(key)
        # print(list_with_single_boxes)
        list_with_all_boxes.append(list_with_single_boxes)
        list_name.append(key)
    return filename, list_with_all_boxes, list_name


def visualize_crop():
    img = cv2.imread("examples/FR_20190307100749_0000541.jpeg")
    x = 200
    y0 = 0
    y1 = 360
    crop_img = img[x:x+340, y0:y0+570]
    cv2.imshow("cropped0", crop_img)

    crop_img = img[x:x+340, y1:y1+570]
    cv2.imshow("cropped1", crop_img)

    crop_img = img[x:x+340, -570:-1]
    cv2.imshow("cropped2", crop_img)
    cv2.imshow("original", img)
    cv2.waitKey(0)
################################


def convert_to_small(bound_original, type = 1):
    xmi = bound_original[0]
    ymi = bound_original[1]
    xma = bound_original[3]
    yma = bound_original[4]
    if type == 1:
        ymi -= 200
        yma -= 200
    elif type == 2:
        ymi -= 200
        yma -= 200
        xmi -= 360
        xma -= 360
    elif type == 3:
        ymi -= 200
        yma -= 200
        xmi -= 710
        xma -= 710


def convert_to_big(bound_small, type = 1):
    xmi = bound_small[0]
    ymi = bound_small[1]
    xma = bound_small[3]
    yma = bound_small[4]
    if type == 1:
        ymi += 200
        yma += 200
    elif type == 2:
        ymi += 200
        yma += 200
        xmi += 360
        xma += 360
    elif type == 3:
        ymi += 200
        yma += 200
        xmi += 710
        xma += 710


def dataframe_generator(source_data):
    files = os.listdir(source_data)
    file_list = []
    xmis = []
    ymis = []
    xmas = []
    ymas = []
    type_boxs = []
    for fi in files:
        if fi[-4:] == 'jpeg':
            pa_fi = os.path.join(source_data, fi)
            xml_fi = fi.replace('jpeg', 'xml')
            pa_xml = os.path.join(source_data, xml_fi)
            boxes = parsexml(pa_xml)
            for box, na in zip(boxes[1], boxes[2]):
                file_list.append(pa_fi)
                xmis.append(box[0])
                ymis.append(box[1])
                xmas.append(box[2])
                ymas.append(box[3])
                type_boxs.append(na)
    frame = pd.DataFrame(list(zip(file_list, xmis, ymis, xmas, ymas, type_boxs)))
    # frame.to_csv('frame.csv', header=False, index=False)
    # print(frame)
    return frame


def csv_gen(list_path):
    list_name = os.listdir(list_path)
    # print(list_name)
    frames = []
    for da in list_name:
        da_path = os.path.join(list_path, da)
        frame = dataframe_generator(da_path)
        frames.append(frame)
    return pd.concat(frames)

# dataframe_generator(r'D:\data\asin_data\191028_variation_cart_toFPT')
out = csv_gen(r'D:\data\asin_data')
print(out)
out.to_csv('frame.csv', header=False, index=False)
