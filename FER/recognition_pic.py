import os # 文件模块
import tensorflow as tf
import cv2 # 图像处理
import numpy as np
import csv # 图表数据处理
import tqdm
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

import recognition

data_folder_name = '..\\data'
data_path_name = 'cv'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
ckpt_name = 'cnn_emotion_classifier.ckpt'
model_path_name = 'cnn_inuse'
casc_name = 'haarcascade_frontalface_alt.xml' # Opencv级联分类器
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)


channel = 1
default_height = 48
default_width = 48
confusion_matrix = False
emotion_labels = ['生气', '尴尬', '害怕', '开心', '伤心', '惊讶', '自然']


def main(path):
    if not confusion_matrix:
        image=path
        print(image+'\n')
        faces, img_gray, img = recognition.face_detect_pic(image)  #识别人脸并转化为灰度图像
        spb = img.shape   #显示尺寸：  宽度，高度，通道数
        sp = img_gray.shape
        height = sp[0]
        width = sp[1]
        size = 600
        emotion_pre_dict = {}
        face_exists = 0
        for (x, y, w, h) in faces:
            face_exists = 1
            face_img_gray = img_gray[y:y + h, x:x + w]
            results_sum = recognition.predict_emotion(face_img_gray)  # face_img_gray
            for i, emotion_pre in enumerate(results_sum):
                emotion_pre_dict[emotion_labels[i]] = emotion_pre
            # 输出所有情绪的概率
            print(emotion_pre_dict)
            label = np.argmax(results_sum)
            emo = emotion_labels[int(label)]
            print('Emotion : ', emo)
            # 输出最大概率的情绪
            # 使框的大小适应各种像素的照片
            with open('../data/cv/1.txt', 'w') as f:
                f.write(str(emotion_pre_dict)+'\n')
                f.write(str('Emotion : '+str(emo)))
            #将所有情绪概率存入文件中，以便后期可视化
            t_size = 2
            ww = int(spb[0] * t_size / 300)
            www = int((w + 10) * t_size / 100)
            www_s = int((w + 20) * t_size / 100) * 2 / 5
            # 用方框勾画出脸部位置
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 205, 211), ww)
            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)  # 图片上打印
            font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
            draw.text((x + 20, y + h+10 ), emo, (255, 205, 211), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
            # PIL图片转cv2 图片
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            #框出人脸并将最大概率的情绪名称注明在图像上
            # img_gray full face     face_img_gray part of face
        if face_exists:
            cv2.imwrite('../data/cv/pic/test.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
            return True
        else:
            return False
            #将识别处理后的图像保存，以便后期显示在系统界面上

    if confusion_matrix:
        with open(csv_path, 'r') as f:
            csvr = csv.reader(f)
            header = next(csvr)
            rows = [row for row in csvr]
            val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
            tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        confusion_images_total = []
        confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        test_set = tst
        total_num = len(test_set)
        for label_image_ in test_set:
            label_ = int(label_image_[0])
            image_ = np.reshape(np.asarray([int(p) for p in label_image_[-1].split()]), [default_height, default_width, 1])
            confusion_images[label_].append(image_)
        recognition.produce_confusion_matrix(confusion_images.values(), total_num)


