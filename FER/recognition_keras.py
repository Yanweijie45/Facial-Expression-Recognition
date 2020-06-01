import os # 文件模块
import tensorflow as tf
import cv2 # 图像处理
import numpy as np
import csv # 图表数据处理
from keras.models import load_model

data_folder_name = '..\\data'
data_path_name = 'cv'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
ckpt_name = 'model-keras.h5'
model_path_name = 'cnn_inuse'
casc_name = 'haarcascade_frontalface_alt.xml' # Opencv级联分类器
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)
ckpt_path = os.path.join(data_folder_name, data_path_name, ckpt_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)


channel = 1
default_height = 48
default_width = 48
confusion_matrix = False
use_advanced_method = True
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)
y_=[]

sess = tf.Session()

model = load_model(ckpt_path)


#图像优化
def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, :], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(cv2.flip(image_, 1), (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs


#进行人脸识别（单脸）
def produce_result(images_):
    images_= np.multiply(np.array(images_), 1. / 255)
    pred_logits_=model.predict(images_, batch_size=1, verbose=1)
    return np.sum(pred_logits_, axis=0)


#进行人脸识别（多脸）
def produce_results(images_):
    results = []
    pred_logits_ = produce_result(images_)
    pred_logits_list_ = np.array(np.reshape(np.argmax(pred_logits_, axis=1), [-1])).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)
    return results, result_decimals

#创建混淆矩阵
def produce_confusion_matrix(images_list_, total_num_):
    total = []
    total_decimals = []
    for ii, images_ in enumerate(images_list_):
        results, result_decimals = produce_results(images_)
        total.append(results)
        total_decimals.append(result_decimals)
        print(results, ii, ":", result_decimals[ii])
        print(result_decimals)
    sum = 0
    for i_ in range(num_class):
        sum += total[i_][i_]
    acc=sum * 100. / total_num_
    print('acc: {:.3f} %'.format(sum * 100. / total_num_))
    print('Using ', ckpt_name)

#转换图片为48*48
def predict_emotion(image_):
    image_ = cv2.resize(image_, (48, 48), interpolation=cv2.INTER_AREA)
    image_ = image_.reshape(1, 48, 48, 1)
    return produce_result(image_)

#识别人脸并转化为灰度图像
def face_detect(image_path, casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        img_ = cv2.imread(image_path) #读入图像
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY) #彩色图像转为灰度图像
        # face detection
        # 利用Opencv级联分类器检测脸部
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))


def main(path):
    if not confusion_matrix:
        image=path
        print(image+'\n')
        faces, img_gray, img = face_detect(image)  #识别人脸并转化为灰度图像
        spb = img.shape   #显示尺寸：  宽度，高度，通道数
        sp = img_gray.shape
        emotion_pre_dict = {}
        face_exists = 0
        for (x, y, w, h) in faces:
            face_exists = 1
            face_img_gray = img_gray[y:y + h, x:x + w]
            results_sum = predict_emotion(face_img_gray)  # face_img_gray
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
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
            cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        www_s, (255, 0, 255), thickness=www, lineType=1)
            #框出人脸并将最大概率的情绪名称注明在图像上
            # img_gray full face     face_img_gray part of face
        if face_exists:
            cv2.imwrite('../data/cv/pic/test.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
            return True
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
        produce_confusion_matrix(confusion_images.values(), total_num)


