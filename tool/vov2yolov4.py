import xml.etree.ElementTree as ET
import os
import os.path as osp
from tqdm import tqdm
file_path = ['Train/Train/Annotations', 'Test/Test/Annotations', 'Val/Val/Annotations']
img_path = ['Train/Train/JPEGImages', 'Test/Test/JPEGImages', 'Val/Val/JPEGImages']
files = ['Train/Train/train.txt', 'Test/Test/test.txt', 'Val/Val/val.txt'] #
for i in range(len(file_path)):
    with open(files[i], 'w') as f:
        for xml in tqdm(os.listdir(file_path[i])):

            img = xml.split('.')[0] + '.jpg'
            f.write(img_path[i]+'/'+img + ' ')
            tree = ET.parse(osp.join(file_path[i], xml))
            root = tree.getroot()
            start = 0
            end = len(root.findall('object')) - 1
            for obj in root.findall('object'):

                bnd_box = obj.find('bndbox')
                x1 = bnd_box.find('xmin').text
                y1 = bnd_box.find('ymin').text
                x2 = bnd_box.find('xmax').text
                y2 = bnd_box.find('ymax').text
                f.write(x1 + ',' + y1 + ',' + x2 + ',' + y2 + ',')
                if obj.find('name').text == 'person':
                  if start != end:
                    f.write('1' + ' ')
                  else:
                    f.write('1')
                elif obj.find('name').text == 'person-like' or obj.find('name').text == 'person-fa':
                  if start != end:
                    f.write('0' + ' ')
                  else:
                    f.write('0')
                else:
                    continue
                start += 1
            f.write('\n')