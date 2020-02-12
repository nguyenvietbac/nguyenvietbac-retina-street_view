import os
import csv


def read_classes(file):
    """ Parse the classes file given by csv_reader.
    """
    name = ['person',#,0
            'bicycle',#,1
            'car',#,2
            'motorcycle',#,3
            'traffic light',#,4
            'traffic sign',#,5
            'parking meter',#,6
            'u-shaped barricade',#,7
            'short pole',#,8
            'safety cone',#,9
            'construction barricade',#,10
            'parking block',#,11
            'shopping cart',#,12
            'tricycle',#,13
            'stroller',#,14
            'wheelchair',#,15
            'other compact ride',#,16
            'locked parking barrier',#,17
            'unlocked parking barrier',#,18
            ]

    result = {}
    for line, row in enumerate(name):
        result[str(line)] = row
    return result


def make_xml_annotation(boxes, scores, labels, img_name, xml_path, confident_threshold):
    zind = 0
    map_label = read_classes('../data/current_object.txt')

    # for z in data:
    basename = img_name.replace('.jpeg', '.xml')
    basename = basename.replace('.png', '.xml')
    f = open(os.path.join(xml_path,  basename),'w')
    line = "<annotation>" + '\n'
    f.write(line)
    line = '\t<folder>' + "folder" + '</folder>' + '\n'
    f.write(line)
    line = '\t<filename>' + img_name + '</filename>' + '\n'
    f.write(line)
    line = '\t<source>\n\t\t<database>Asin</database>\n\t</source>\n'
    f.write(line)
    line = '\t<size>\n\t\t<width>'+ '1280' + '</width>\n\t\t<height>' + '800' + '</height>\n\t'
    line += '\t<depth>Unspecified</depth>\n\t</size>'
    f.write(line)
    line = '\n\t<segmented>Unspecified</segmented>'
    f.write(line)
    ind = 0
    for bx, sc, la in zip(boxes, scores, labels):
        if sc < confident_threshold:
            continue
        # print(la)
        line = '\n\t<object>'
        line += '\n\t\t<name>' + str(map_label[(str(la))]) + '</name>\n\t\t<pose>Unspecified</pose>'
        line += '\n\t\t<truncated>Unspecified</truncated>\n\t\t<difficult>' + str(sc) + '</difficult>'
        xmin = bx[0]
        # print(bx)
        # print(bx[0])
        line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(int(xmin)) + '</xmin>'
        ymin = bx[1]
        line += '\n\t\t\t<ymin>' + str(int(ymin)) + '</ymin>'
        xmax = bx[2]
        ymax = bx[3]
        line += '\n\t\t\t<xmax>' + str(int(xmax)) + '</xmax>'
        line += '\n\t\t\t<ymax>' + str(int(ymax)) + '</ymax>'
        line += '\n\t\t</bndbox>'
        line += '\n\t</object>\n'
        f.write(line)
        ind += 1
    line += '</annotation>\n'
    f.write(line)
    f.close()
