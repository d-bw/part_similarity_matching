import yaml
import cv2
import numpy as np
from PIL import Image
#from google.colab.patches import cv2_imshow
# 读取yaml文件
import argparse
from feature_extract import remove_duplicate_contours, get_bitImage_byContours


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


if __name__=='__main__':



  parser = argparse.ArgumentParser(description='choose for yaml file')

  parser.add_argument('-file', '--file_name', dest='file', type=str,required=True,
                     help='input your yaml path')
  args = parser.parse_args() 

  #file_name='../test_resource/test.yaml'
  with open(args.file, 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)


  contours,hierarchy=extract_contours(data,42)
  #contours,hierarchy=remove_duplicate_contours(contours,hierarchy)

  image=get_bitImage_byContours(contours,hierarchy)

  image.save("show_yaml.jpg")

  

  



    