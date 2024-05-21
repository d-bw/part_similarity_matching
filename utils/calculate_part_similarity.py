import cv2
from transformers import pipeline
import torch
import yaml
from preprocess_yaml import extract_contours
from feature_extract import (preprocess_img_byRmBackground,
                extract_main_object_mask, 
                get_contours_on_image,  
                get_bitImage_byContours,
                align_contours,
                show_mask,
                find_best_rotation
                )


from statistics import mean
import numpy as np
import matplotlib.pyplot as plt




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


# orb特征点提取，对比相似度
def calculate_orb(image1,image2):
  image1 = np.array(image1)
  image2 = np.array(image2)

  orb = cv2.ORB_create(nfeatures=2048)
  keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
  keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

  if descriptors1 is None or descriptors2 is None:
      #print("Error: Failed to extract descriptors.")
      return "nav"

  #BFM特征匹配器
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(descriptors1, descriptors2)

  matches = sorted(matches, key=lambda x: x.distance)
  match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  score = sum([match.distance for match in matches]) / len(matches)


  # FLANN匹配器参数
#   FLANN_INDEX_LSH = 6
#   index_params = dict(algorithm=FLANN_INDEX_LSH,
#                       table_number=6,
#                       key_size=12,
#                       multi_probe_level=1)
#   search_params = dict(checks=50)

#   # 创建FLANN匹配器
#   flann = cv2.FlannBasedMatcher(index_params, search_params)

#   # 使用FLANN匹配器进行特征匹配
#   matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#   print(len(matches))
# # 应用比率测试以保留良好的匹配
#   good_matches = []
#   for match_pair in matches:
#     if len(match_pair) < 2:  # 检查是否有足够的匹配项
#         continue
#     m, n = match_pair
#     if m.distance < 0.7 * n.distance:
#         good_matches.append(m)

#   match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  #score = sum([match.distance for match in good_matches]) / len(good_matches)        
  cv2.imwrite('show_features.jpg',match_img) 
  return score


#最大面积计算相似度
def calculate_max_overlapArea(contours1, contours2, hierarchy1, hierarchy2, image1, image2):
  contours1, hierarchy1, first_rotate_angle = align_contours(contours1, contours2, hierarchy1, hierarchy2)
  #print(first_rotate_angle)
  
  #image1.save('show_segment.jpg')

  best_angle, best_overlap ,rotated_contour1,hierarchy1=find_best_rotation(contours1, hierarchy1, image1,image2)
  # print(best_angle, best_overlap)

  image1=get_bitImage_byContours(rotated_contour1,hierarchy1)
 
  # image1.save('show_segment.jpg')
  # image2.save("show_yaml.jpg")

  return (best_angle+first_rotate_angle)%360, best_overlap







#处理照片零件
device= "cuda" if torch.cuda.is_available() else "cpu"


generator =  pipeline("mask-generation", model="facebook/sam-vit-huge", device = device, points_per_batch = 256)

preprocessed_image=preprocess_img_byRmBackground("./test_resource/test142.jpg")





outputs = generator(preprocessed_image, points_per_batch = 256)

ax = plt.gca()
plt.axis("off")
plt.imshow(preprocessed_image)

for mask in outputs["masks"]:
  show_mask(mask,ax)

main_object_mask = extract_main_object_mask(outputs["masks"])

contours1,hierarchy1 = get_contours_on_image(main_object_mask)
image1=get_bitImage_byContours(contours1,hierarchy1)




#处理图纸零件
with open("./test_resource/test.yaml", 'r') as file:
  data = yaml.load(file, Loader=yaml.FullLoader)


for i in range(len(data['parts'])):

  contours2,hierarchy2=extract_contours(data,i)
  
  image2=get_bitImage_byContours(contours2,hierarchy2)
  # image1 = cv2.cvtColor(cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
  # image2 = cv2.cvtColor(cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
  
  _ , ratio = calculate_max_overlapArea(contours1, contours2, hierarchy1, hierarchy2, image1, image2)
  if ratio >0.8:  
    print(f"该零件与图纸第{i}个零件的相似度为{ratio}")
 


# contours2,hierarchy2=extract_contours(data,33)

# image2=get_bitImage_byContours(contours2,hierarchy2)
# #contours1, hierarchy1, angle =align_contours(contours1, contours2, hierarchy1, hierarchy2)
# # print(angle)
# # image1=get_bitImage_byContours(contours1,hierarchy1)
# # image1.save('show_segment.jpg')
# # image2.save("show_yaml.jpg")

# print(calculate_max_overlapArea(contours1, contours2, hierarchy1, hierarchy2, image1, image2))








