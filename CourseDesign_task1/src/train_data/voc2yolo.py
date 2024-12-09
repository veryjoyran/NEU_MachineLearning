import os
import xml.etree.ElementTree as ET
import logging
import shutil
import json

IMAGESPATH = 'JPEGimages'
distribution_rate = 0.80 # 按照分配率将图片分类 分类率：train/validation 可以自己修改，可以不改，看心情
# ↓↓标注的类别，很重要很重要，这里必须要改，按照自己的类别去改|
classes = ['Fovea']

# 提取图像并将其分类为训练集与训练测试集 ：train  validation
Images = os.listdir(IMAGESPATH)
Images = [i for i in Images if i.split('.')[-1] == 'jpg']
print('提取到有效jpg图片共{}张'.format(len(Images)))



# 正式移动图片到指定目录：.images 下, 并且生成训练索引 train.txt and val.txt 这一步会清空这两个文本的内容
# 正式移动图片到指定目录：.images 下, 并且生成训练索引 train.txt and val.txt 这一步会清空这两个文本的内容
train = Images[0: int(distribution_rate*len(Images))]
validation = Images[int(distribution_rate*len(Images)):]
if train == 0 or validation == 0:
    raise FileExistsError('没有找到训练集的图片或测试集图片，请检查目录')

# 获取绝对路径。为了好看 把 \ 处理成 /
# ab_path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
# print(ab_path)
dest_path = 'dataset_test'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

train_label_path = os.path.join(dest_path,'labels/train')
validation_label_path = os.path.join(dest_path,'labels/val')
train_image_path = os.path.join(dest_path,'images/train')
validation_image_path = os.path.join(dest_path,'images/val')
# 生成相关目录
for eachPath in [train_label_path,validation_label_path,train_image_path,validation_image_path]:
    if not os.path.exists(eachPath):
        os.makedirs(eachPath)

with open(os.path.join(dest_path,'train.txt'), 'w', encoding='utf-8') as f:
    for i in train:
        f.write('../'+dest_path + '/images/train/' + i + '\n')
        shutil.copy(IMAGESPATH+'/' + i, os.path.join(dest_path,'images/train') )

with open(os.path.join(dest_path,'val.txt'), 'w', encoding='utf-8') as f:
    for i in validation:
        f.write('../'+dest_path + '/images/val/' + i + '\n')
        shutil.copy(IMAGESPATH +'/' + i, os.path.join(dest_path,'images/val'))

print('图片移动/复制完成，训练索引 train.txt and val.txt 生成完毕')


# 预检测 xml与图片的对应关系，这里要求严格一一对应


xml_file = os.listdir('Annotations')
xml_file = [i for i in xml_file if i.split('.')[-1] == 'xml']
xml_file.sort(key=lambda x: int(x.split('.')[0][1:]))
# print(xml_file)
xml_file_check = [i.split('.')[0]+'.xml' for i in Images if i.split('.')[-1] == 'jpg']
xml_file_check.sort(key=lambda x: int(x.split('.')[0][1:]))
# print(xml_file_check)
if xml_file_check != xml_file:
    raise FileExistsError('Annotations 中xml文件与JPEGImages图片不对应，请仔细检测！')
# 下面将 xml文件标注提取并生成label
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def write_labels(xml_file_path, write_to_file_path):
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        with open(write_to_file_path, 'w', encoding='utf-8') as f2:
            for obj in root.iter('object'):
                xml_name = obj.find('name').text
                if xml_name not in classes:
                    logging.warning('正在检索该对象不存在设定classes，应该引起重视')
                    continue
                cls_id = classes.index(xml_name)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                b1, b2, b3, b4 = b
                # 标注越界修正
                if b2 > w:
                    b2 = w
                if b4 > h:
                    b4 = h
                b = (b1, b2, b3, b4)
                bb = convert((w, h), b)
                write_message = str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                f2.write(write_message)
                if not write_message:
                    logging.warning('未在标注图片的xml文件中取得分类内容，此警告应引起重视，可能意味着分类参数不匹配。classes错误')


for i in train:
    write_labels('Annotations/' + i.split('.')[0] + '.xml', os.path.join(dest_path,'labels/train/{}'.format(i.split('.')[0] + '.txt')))

for i in validation:
    write_labels('Annotations/' + i.split('.')[0] + '.xml', os.path.join(dest_path,'labels/val/{}'.format(i.split('.')[0] + '.txt')))

# 生成yaml文件
yaml_content= f'''train: ./train.txt
val: ./val.txt

nc: {len(classes)}
names: {json.dumps(classes)}
'''

with open(os.path.join(dest_path,dest_path+'.yaml'),'w') as f:
    f.write(yaml_content)

print('finish work')