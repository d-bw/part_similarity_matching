import cv2
from transformers import pipeline
import torch
import yaml
from preprocess_yaml import extract_contours
from feature_extract import (preprocess_img_byRmBackground,
                extract_main_object_mask, 
                get_contours_on_image, 
                remove_duplicate_contours, 
                get_bitImage_byContours)
from statistics import mean
import numpy as np


#简单Hu矩对比最外层轮廓，不考虑孔洞——适用于简单情况
def simple_HuMoment(image1,image2):
  #将PIL格式转为open_cv格式
  image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_BGR2GRAY)
  image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2GRAY)
  _, thresh1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
  _, thresh2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
  #提取外边界
  contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #找到最大轮廓
  max_contour1 = max(contours1, key=cv2.contourArea)
  max_contour2 = max(contours2, key=cv2.contourArea)

  similarity = cv2.matchShapes(max_contour1, max_contour2, cv2.CONTOURS_MATCH_I1, 0)
  return 1 / (1 + similarity)


#平均Hu矩对比所有轮廓
def avg_HuMoment(contours1,contours2):
  hu_moments_list1 = []
  """计算contours集合的平均Hu矩"""
  for contour in contours1:
      M = cv2.moments(contour)
      hu_moments = cv2.HuMoments(M)
      hu_moments_list1.append(hu_moments.flatten())

  # 计算contours2中每个轮廓的Hu矩
  hu_moments_list2 = []
  for contour in contours2:
      M = cv2.moments(contour)
      hu_moments = cv2.HuMoments(M)
      hu_moments_list2.append(hu_moments.flatten())

  # 计算平均Hu矩
  average_hu_moments1 = np.mean(hu_moments_list1, axis=0)
  average_hu_moments2 = np.mean(hu_moments_list2, axis=0)
  #print(average_hu_moments1)

  # 使用matchShapes函数比较两个平均Hu矩的相似度
  similarity = np.linalg.norm(average_hu_moments1 - average_hu_moments2)
  return 1 / (1 + similarity)



#处理照片零件
device= "cuda" if torch.cuda.is_available() else "cpu"


generator =  pipeline("mask-generation", model="facebook/sam-vit-base", device = device, points_per_batch = 256)

preprocessed_image=preprocess_img_byRmBackground("./test_resource/test17.jpg")

outputs = generator(preprocessed_image, points_per_batch = 256)


main_object_mask = extract_main_object_mask(outputs["masks"])

contours1,hierarchy1 = get_contours_on_image(main_object_mask)


image1=get_bitImage_byContours(contours1,hierarchy1)


#处理图纸零件
with open("./test_resource/test.yaml", 'r') as file:
  data = yaml.load(file, Loader=yaml.FullLoader)


for i in range(len(data['parts'])):

  contours2,hierarchy2=extract_contours(data,i)
  image2=get_bitImage_byContours(contours2,hierarchy2)
  if avg_HuMoment(contours1,contours2) > 0.998:
    print(f"该零件与图纸第{i}个零件的相似度为{avg_HuMoment(contours1,contours2)}")
  







'''
sift = cv2.SIFT_create()
image1 = np.array(image1)
flann = cv2.FlannBasedMatcher()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
  
for i in range(len(data['parts'])):
  contours2,hierarchy2=extract_contours(data,i)

  image2=get_bitImage_byContours(contours2,hierarchy2)
  image2 = np.array(image2)
  # 检测特征点和计算描述符

  keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

  # 创建FLANN匹配器

  if descriptors1 is not None and descriptors2 is not None:
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # 根据Lowe's ratio test选择好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 计算相似度
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))
    print(f"index:{i}----Similarity: {similarity}")
  else: 
    print(f"index:{i}----Similarity:0")

'''





