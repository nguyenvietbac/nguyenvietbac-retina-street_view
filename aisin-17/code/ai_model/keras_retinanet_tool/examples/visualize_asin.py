import cv2
import xml
import xml.etree.ElementTree as ET
# doc = xml.dom.minidom.parse('D:\project\visual_processing\data\FR_20190214140147_0000040.xml')
# print(doc)
# print(doc.nodeName)


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


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    print(list_with_all_boxes)
    return filename, list_with_all_boxes

# name, boxes = read_content("file.xml")


# pa = 'FR_20190214140147_0000040.xml'
# img = 'FR_20190214140147_0000040.jpeg'
# pa = 'FR_20190703194758_0000060.xml'
# img = 'FR_20190703194758_0000060.jpeg'
# pa = 'FR_20190305140226_0000361.xml'
# img = 'FR_20190305140226_0000361.jpeg'
pa = 'FR_20180409135610_0000013.xml'
img = 'FR_20180409135610_0000013.jpeg'
image = cv2.imread(img)
# read_content(pa)
n, ba, na = parsexml(pa)
# print(b)
count = 0
for b, n in zip(ba, na):
    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
    cv2.imshow('demo', image)
    cv2.waitKey(0)
    count += 1
    print(count)
