import yaml
import cv2
import numpy as np
from PIL import Image
import os 

#from google.colab.patches import cv2_imshow
# 读取yaml文件
import argparse
from feature_extract import remove_duplicate_contours, get_bitImage_byContours
from feature_extract import preprocess_img_byRmBackground, preprocess_img_byCanny

def extract_contours(data,index):

  
  part = data['parts'][index]  # 获取第一个零件的数据
  outside_loop = part['outsideLoop']
  inside_loops = part['insideLoops']

  # 计算零件的长宽
  x_coords = [point['x'] for point in outside_loop]
  y_coords = [point['y'] for point in outside_loop]
  x_coords += [point['x'] for loop in inside_loops for point in loop]
  y_coords += [point['y'] for loop in inside_loops for point in loop]

  min_x, min_y = min(x_coords), min(y_coords)
  max_x, max_y = max(x_coords), max(y_coords)
  part_width = max_x - min_x
  part_height = max_y - min_y

  # 计算放大比例
  target_width = part_width * 2
  target_height = part_height * 2

  scale_factor_width = target_width / part_width
  scale_factor_height = target_height / part_height

  # 选择较小的比例作为放大倍数
  scale_factor = min(scale_factor_width, scale_factor_height)

  # 创建新的画布
  scaled_part_width = int(part_width * scale_factor)+100
  scaled_part_height = int(part_height * scale_factor)+100
  image_resized = np.zeros((scaled_part_height, scaled_part_width), dtype=np.uint8)
  pts = np.array([(int((point['x'] - min_x) * scale_factor), int((point['y'] - min_y) * scale_factor)) for point in outside_loop], dtype=np.int32)
  cv2.fillPoly(image_resized, [pts], color=255)
  # 填充内部轮廓为黑色
  for loop in inside_loops:
      pts = np.array([(int((point['x'] - min_x) * 2), int((point['y'] - min_y) * 2)) for point in loop], dtype=np.int32)
      cv2.fillPoly(image_resized, [pts], color=0)

  # 查找轮廓
  contours,hierarchy = cv2.findContours(image_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  #cv2.imwrite('test.jpg',image_resized)  
  return contours,hierarchy

def merge_images(images, desired_size):

    # 计算总共的图像数量
    num_images = len(images)
    
    # 计算每行、每列可以容纳的图像数量
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    
    # 计算每个小图的正方形尺寸
    max_size = min(desired_size) // max(num_rows, num_cols)
    
    # 缩放图像大小为正方形
    resized_images = [cv2.resize(img, (max_size, max_size)) for img in images]
    
    # 创建空白图像，用于存储拼接后的图像
    blank_image = np.zeros((desired_size[0], desired_size[1]), dtype=np.uint8)
    
    # 计算每个小图的位置并将其拼接到空白图像上
    for i, img in enumerate(resized_images):
        row = i // num_cols
        col = i % num_cols
        y_offset = row * max_size
        x_offset = col * max_size
        blank_image[y_offset:y_offset+max_size, x_offset:x_offset+max_size] = img
    
    return blank_image


def make_dataset(input_folder, output_folder):
  # 如果输出文件夹不存在，则创建
  if not os.path.exists(output_folder):
      os.makedirs(output_folder)

  # 获取输入文件夹中所有文件的列表
  files = os.listdir(input_folder)

  # 初始化文件编号
  file_number = 0

  # 遍历所有文件
  for file in files:
    # 检查文件是否是pil格式的图像
    if file.lower().endswith('.jpg'):
      # 打开图像
      img_path = os.path.join(input_folder, file)
      with Image.open(img_path) as img:
        # 缩放图像至224*224
        pic = preprocess_img_byRmBackground(img.filename)
        pic= pic.convert('L')

        img = pic.resize((224, 224), Image.ANTIALIAS)
        # 生成输出文件名
        output_file_name = f"{file_number}.bmp"
        output_file_path = os.path.join(output_folder, output_file_name)
        # 保存为bmp格式
        img.save(output_file_path, format='BMP')
        # 增加文件编号
        file_number += 1


if __name__=='__main__':



  parser = argparse.ArgumentParser(description='choose for yaml file')

  parser.add_argument('-file', '--file_name', dest='file', type=str,required=True,
                    help='input your yaml path')
  args = parser.parse_args() 
  #预处理操作
  # #file_name='../test_resource/test.yaml'
  # with open(args.file, 'r') as file:
  #   data = yaml.load(file, Loader=yaml.FullLoader)


  # contours,hierarchy=extract_contours(data,36)
  # #contours,hierarchy=remove_duplicate_contours(contours,hierarchy)
  # #print(contours[0])
  # image=get_bitImage_byContours(contours,hierarchy)

  # image.save("show_yaml.jpg")
  #数据集构建操作
  input_folder = '/content/drive/MyDrive/part_similarity_matching/dataset/test_dataset/PART/part_matching/5'  # 替换为实际的输入文件夹路径
  output_folder = '/content/drive/MyDrive/part_similarity_matching/dataset/test_dataset/PART/part_matching/5/reference/1'  # 替换为实际的输出文件夹路径
  make_dataset(input_folder, output_folder)
  #合并图像操作
  # with open(args.file, 'r') as file:
  #   data = yaml.load(file, Loader=yaml.FullLoader)

  # image_list=[]
  # for i in range (len(data['parts'])):

  #   contours,hierarchy=extract_contours(data,i)
  #   #contours,hierarchy=remove_duplicate_contours(contours,hierarchy)

  #   image=get_bitImage_byContours(contours,hierarchy)
  #   image=cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
  #   image_list.append(image)

  # big_image=merge_images(image_list,desired_size = (800, 800))
  # cv2.imwrite('big_image.jpg',big_image)



  




  

  



    