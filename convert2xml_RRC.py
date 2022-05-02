# convert the kitti type txt-format labels to voc type xml-format labels -- to dir structure suitable for RRC training.
import os
import glob
from PIL import Image
from lxml import etree
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--dataset-name', help='name of the dataset to be processed', type=str, default='KITTI')
args = parser.parse_args()

root_dir = os.path.expanduser('~') + '/data/' + args.dataset_name + '/'
img_type = '.png'
img_dir = root_dir + 'training/image_2/'
txt_dir = root_dir + 'training/label_2/'
xml_dir = root_dir + 'training/label_2/xml/'

def dict2annotation(d, parent=None):
    if parent is None:
        parent = etree.Element('annotation')
    for key, value in d.items():
        if isinstance(value, str):
            element = etree.SubElement(parent, key)
            element.text = value
        elif isinstance(value, int):
            element = etree.SubElement(parent, key)
            element.text = str(int(value))
        elif isinstance(value, float):
            element = etree.SubElement(parent, key)
            element.text = str(float(value))
        elif isinstance(value, dict):
            element = etree.SubElement(parent, key)
            dict2annotation(value, element)
        elif isinstance(value, list):
            for text in value:
                element = etree.SubElement(parent, key)
                dict2annotation(text, element)
        else:
            raise TypeError('Unexpected value type: {0}'.format(type(value)))
    return parent

classNames = ['Car', 'Pedestrian', 'Cyclist', 'HeadlightLeft', 'HeadlightRight', 'TaillightLeft', 'TaillightRight', 'StopLine']

if not os.path.exists(xml_dir):
    os.makedirs(xml_dir)

txtfilepath_list = glob.glob(txt_dir + "/*.txt")
for txtfilepath in txtfilepath_list:
    file_idx = os.path.split(txtfilepath)[-1].split('.')[0]
    img = Image.open(img_dir + file_idx + img_type)
    vocAnnotation = {
        'folder': 'training/image_2/',
        'filename': file_idx + img_type,
        'size': {
            'width': img.size[0],
            'height': img.size[1],
            'depth': 3
        },
        'object': []
    }
    txtfile = open(txtfilepath, 'r')
    lines = [i.strip() for i in txtfile.readlines()]
    for line in lines:
        content = line.split(' ')
        if content[0] not in classNames:
            continue
        vocAnnotation['object'].append(
            {
                'name': content[0],
                'truncated': int(float(content[1])),
                'occluded': int(content[2]),
                'alpha': float(content[3]),
                'bndbox': {
                    'xmin': float(content[4]),
                    'xmax': float(content[6]),
                    'ymin': float(content[5]),
                    'ymax': float(content[7])
                },
                'dimensions': {
                    'height': float(content[8]),
                    'width' : float(content[9]),
                    'length': float(content[10])
                },
                'location': {
                    'x': float(content[11]),
                    'y': float(content[12]),
                    'z': float(content[13])
                },
                'rotation_y': float(content[14])
            }
        )
    xml_file = open(xml_dir + file_idx + '.xml', 'w')
    xml_file.write(etree.tostring(dict2annotation(vocAnnotation), encoding="unicode", pretty_print=True))
