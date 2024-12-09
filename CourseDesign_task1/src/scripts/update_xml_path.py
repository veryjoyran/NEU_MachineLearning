import os
import xml.etree.ElementTree as ET

# 设置新文件夹路径
new_folder_name = "Annotations"  # folder字段修改为的名字
new_base_path = "./CourseDesign_task1/src/train_data/JPEGimages"  # 新的图像路径的根目录

# 假设XML文件存放在该路径
xml_folder = "./CourseDesign_task1/src/train_data/Annotations"
image_folder = "./CourseDesign_task1/src/train_data/JPEGimages"

# 获取该文件夹下所有的xml文件
xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]

# 按文件名中的数字排序
xml_files.sort(key=lambda x: int(x.split('.')[0][1:]))  # 假设文件名格式为xxxx.xml，如0001.xml

# 遍历所有XML文件并修改folder和path标签
for xml_file in xml_files:
    # 构建XML文件的完整路径
    xml_path = os.path.join(xml_folder, xml_file)

    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 修改<folder>标签
    folder_tag = root.find('folder')
    if folder_tag is not None:
        folder_tag.text = new_folder_name  # 修改folder的内容

    # 修改<path>标签
    path_tag = root.find('path')
    if path_tag is not None:
        # 从XML的filename标签获取图像的文件名
        filename_tag = root.find('filename')
        if filename_tag is not None:
            image_filename = filename_tag.text  # 获取图像文件名（例如: 0001.png）

            # 构建新的图像路径
            new_image_path = os.path.join(new_base_path, image_filename)
            path_tag.text = new_image_path  # 修改path为新的路径

    # 保存修改后的XML文件
    tree.write(xml_path)

    print(f"Updated {xml_file}")

print("All XML files have been updated.")
