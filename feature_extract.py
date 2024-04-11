import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#from google.colab.patches import cv2_imshow
import argparse
from transformers import pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity


parser = argparse.ArgumentParser(description='choose for show features on the image')

parser.add_argument('-model_id', '--id', dest='model_id', type=str,required=True,
                     help='input your model name space')
parser.add_argument('-image_path', '-path', dest='image_path', type=str,required=True,
                     help='input the absolute path of your image')
args = parser.parse_args()

#提取主要物体mask
def extract_main_object_mask(masks):
    # 计算每个mask的面积，并将其与对应的mask一起存储在一个列表中
    masks_with_areas = [(mask, np.sum(mask)) for mask in masks]

    # 使用面积作为关键字进行排序
    sorted_masks = sorted(masks_with_areas, key=lambda x: x[1], reverse=True)

    # 仅返回排序后的masks，不包括面积信息
    masks = [mask[0] for mask in sorted_masks]

    # 使用第一个mask作为初始主要物体mask
    main_object_mask = masks[0].astype(np.uint8)

    # 遍历剩余的mask，将其作为孔洞剔除
    for mask in masks[1:]:
        # 使用cv2.subtract从主要物体mask中去除孔洞
        main_object_mask = cv2.subtract(main_object_mask, mask.astype(np.uint8))

    return main_object_mask

#展示mask效果
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

#获取分割边界轮廓
def get_contours_on_image(raw_image, mask):


    mask_binary = mask.astype(np.uint8) * 255

    # 图像缩放
    scale_percent = 50
    width = int(mask_binary.shape[1] * scale_percent / 100)
    height = int(mask_binary.shape[0] * scale_percent / 100)
    dim = (width, height)

    mask_binary = cv2.resize(mask_binary, dim, interpolation=cv2.INTER_AREA)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)


    for contour in contours:
      # 获取bounding box坐标
      x, y, w, h = cv2.boundingRect(contour)

      # 画出bounding box
      #cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 红色

      # 计算长、宽
      length = w
      width = h
      cv2.drawContours(annotated_image, [contour], -1, (0, 0, 255), 2)  # 红色，线宽为2
      # 输出bounding box坐标和长宽
      #print(f"Bounding Box: (x:{x}, y:{y}, w:{w}, h:{h}), Length: {length}, Width: {width}")

    # 将布尔类型的掩码转换为整数类型
    mask = mask.astype(np.uint8)

    #cv2.imshow('contours of main part',annotated_image)

    #cv2.waitKey(0)

    #cv2.destroyAllWindows()
    #cv2_imshow(annotated_image)

    #cv2.imwrite("part_annotated_image.jpg", annotated_image)
    return contours

#计算特征
def calculate_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


#特征提取
def get_features(contours):
  features = []
  for contour in contours:
    hu_moments = calculate_hu_moments(contour)
    features.append(hu_moments)

  features = np.array(features)
  return features


'''
#计算余弦相似度
def calculate_cosine_similarity(feature1, feature2):
    # 将特征向量转换为1xN的形状
    feature1 = np.reshape(feature1, (1, -1))
    feature2 = np.reshape(feature2, (1, -1))

    return cosine_similarity(feature1, feature2)[0][0]
'''


if __name__=='__main__':

  device= "cuda" if torch.cuda.is_available() else "cpu"

  generator =  pipeline("mask-generation", model=args.model_id, device = device, points_per_batch = 256)
  
  image_url = args.image_path
  outputs = generator(image_url, points_per_batch = 256)
  raw_image = Image.open(args.image_path,'r')
  #plt.imshow(np.array(raw_image))
  #ax = plt.gca()
  #plt.axis("off")
  main_object_mask = extract_main_object_mask(outputs["masks"])
  #show_mask(main_object_mask, ax=ax)
  #plt.show()
  contours=get_contours_on_image(raw_image,main_object_mask)
  features = get_features(contours)
  print(features)
